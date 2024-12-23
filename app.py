from flask import Flask, request, jsonify, render_template
import re
from collections import Counter
import numpy as np
import javalang
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
import time

def run_java_code_with_input(java_code: str, inputs: str, expected_output: str, timeout: int = 10):
    class_name_match = re.search(r'public\s+class\s+(\w+)', java_code)
    if not class_name_match:
        return "No public class found in the code."
    class_name = class_name_match.group(1)

    # Buat direktori sementara
    temp_dir = tempfile.mkdtemp()
    java_filename = os.path.join(temp_dir, f"{class_name}.java")
    class_filename = os.path.join(temp_dir, f"{class_name}.class")

    try:
        # Tulis file kode Java
        with open(java_filename, "w", encoding="utf-8") as file:
            file.write(java_code)

        # Compile kode Java
        compile_process = subprocess.run(
            ["javac", java_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if compile_process.returncode != 0:
            return f"Compilation Error:\n{compile_process.stderr.strip()}"

        # Jalankan kode Java dengan input
        try:
            run_process = subprocess.run(
                ["java", "-cp", temp_dir, class_name],
                input=inputs,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return "Runtime Error: The program took too long to execute."

        if run_process.returncode != 0:
            return f"Runtime Error:\n{run_process.stderr.strip()}"

        # Validasi output
        actual_output = run_process.stdout.strip()
        # Return langsung expected apa actual output
        return "Actual output : " + actual_output + ", expected output : " + expected_output

    except Exception as e:
        return f"Error occurred: {str(e)}"

    finally:
        # Bersihkan file dan direktori sementara
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except PermissionError as e:
            print(f"PermissionError while deleting temp files: {e}")
            time.sleep(1)  # Beri waktu sebelum mencoba lagi
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
def token_positions(tokens, text):
    positions = []
    for token in tokens:
        start = text.find(token)
        if start != -1:
            positions.append((token, start, start + len(token)))
    return positions

def clean_code(code):
    code = re.sub(r'//.*', '', code)  # Hapus komentar baris
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)  # Hapus komentar blok
    code = re.sub(r'import\s+.*;', '', code)  # Hapus import
    code = re.sub(r'\s+', ' ', code).strip()  # Gabungkan whitespace
    return code

def tokenize_code(cleaned_code):
    try:
        tokens = list(javalang.tokenizer.tokenize(cleaned_code))
        return [token.value for token in tokens]
    except (javalang.tokenizer.LexerError, javalang.parser.JavaSyntaxError):
        return []

def generate_n_grams(tokens, n=4):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def highlight_syntax(tokens1, tokens2):
    highlighted_tokens = []
    for token in tokens1:
        if token in tokens2:
            highlighted_tokens.append(token)
    return highlighted_tokens

def cosine_similarity(ngrams1, ngrams2):
    unique_ngrams = list(set(ngrams1).union(set(ngrams2)))
    freq_ngrams1 = Counter(ngrams1)
    freq_ngrams2 = Counter(ngrams2)

    tf1 = np.array([freq_ngrams1.get(ng, 0) for ng in unique_ngrams])
    tf2 = np.array([freq_ngrams2.get(ng, 0) for ng in unique_ngrams])

    idf = np.log((2 / (np.array([ng in ngrams1 for ng in unique_ngrams]) +
                       np.array([ng in ngrams2 for ng in unique_ngrams])))) + 1

    tfidf1 = tf1 * idf
    tfidf2 = tf2 * idf

    dot_product = np.dot(tfidf1, tfidf2)
    magnitude1 = np.sqrt(np.dot(tfidf1, tfidf1))
    magnitude2 = np.sqrt(np.dot(tfidf2, tfidf2))

    return (dot_product / (magnitude1 * magnitude2)) * 100 if magnitude1 and magnitude2 else 0

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    code1 = request.form.get("code1", "")
    code2 = request.form.get("code2", "")
    expected_output = request.form.get("expected_output", "")
    inputs = request.form.get("inputs", "")

    code1_clean = clean_code(code1)
    code2_clean = clean_code(code2)

    tokens1 = tokenize_code(code1_clean)
    tokens2 = tokenize_code(code2_clean)

    if not tokens1 or not tokens2:
        return jsonify({"error": "Tokenization failed. Please check the syntax of your code."})

    n = 4
    n_grams_1 = generate_n_grams(tokens1, n)
    n_grams_2 = generate_n_grams(tokens2, n)

    similarity = cosine_similarity(n_grams_1, n_grams_2)

    highlighted_tokens = highlight_syntax(tokens1, tokens2)
    common_positions1 = token_positions(highlighted_tokens, code1_clean)
    common_positions2 = token_positions(highlighted_tokens, code2_clean)

    with ThreadPoolExecutor() as executor:
        future1 = executor.submit(run_java_code_with_input, code1, inputs, expected_output)
        future2 = executor.submit(run_java_code_with_input, code2, inputs, expected_output)

        java_output1 = future1.result()
        java_output2 = future2.result()

    is_similar = "output and input same"

    if java_output1 != java_output2:
        is_similar = "output and input different"

    return jsonify({
        "similarity": round(similarity, 2),
        "is_similar": is_similar,
        "common_tokens": list(set(highlighted_tokens)),
        "common_positions_code1": common_positions1,
        "common_positions_code2": common_positions2,
        "clean_code1": code1_clean,
        "clean_code2": code2_clean,
        "java_output_code1": java_output1,
        "java_output_code2": java_output2
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
