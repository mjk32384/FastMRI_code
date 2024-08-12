import subprocess
import sys

result = subprocess.check_output("python add35.py", shell = True)
print("Done")
print(result)