import subprocess
import os

def run_java_code_with_input(java_code: str, inputs: str, expected_output: str):
    java_filename = "TempJavaCode.java"
    class_filename = "TempJavaCode.class"

    with open(java_filename, "w") as file:
        file.write(java_code)

    try:
        compile_process = subprocess.run(
            ["javac", java_filename],
            capture_output=True,
            text=True
        )

        if compile_process.returncode != 0:
            return f"Compilation Error:\n{compile_process.stderr}"

        run_process = subprocess.run(
            ["java", java_filename.replace(".java", "")],
            input=inputs,
            capture_output=True,
            text=True
        )

        if run_process.returncode != 0:
            return f"Runtime Error:\n{run_process.stderr}"

        actual_output = run_process.stdout.strip()
        if actual_output == expected_output.strip():
            return f"Output is correct: {actual_output}"
        else:
            return (
                f"\u274C Output mismatch:\n"
                f"Expected:\n{expected_output}\n"
                f"Actual:\n{actual_output}\n"
                f"Please review the logic or expected output."
            )

    finally:
        if os.path.exists(java_filename):
            os.remove(java_filename)
        if os.path.exists(class_filename):
            os.remove(class_filename)


# java_code = """
# import java.util.Scanner;

# public class TempJavaCode {
#     public static void main(String[] args) {
#         // Output pertama
#         System.out.println("Hello, World!");

#         // Output kedua
#         System.out.println("Enter a number:");

#         Scanner scanner = new Scanner(System.in);
#         int number = scanner.nextInt();

#         // Output ketiga
#         System.out.println("Square of the number is: " + (number * number));

#         scanner.close(); // Tutup scanner untuk menghindari kebocoran sumber daya
#     }
# }
# """
# inputs = "20\n"
# expected_output = "Hello, World!\nEnter a number:\nSquare of the number is: 400"
# output = run_java_code_with_input(java_code, inputs, expected_output)
# print(output)
