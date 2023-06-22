# Import the os module to work with files
import os
import shutil
# Define a list of file names
files = ["config1.ini", "config2.ini", "config3.ini", "config4.ini", "config5.ini"]

# Ask the user to enter a number between 1 and 5
n = int(input("Select between Config 1 and 5: "))

# Check if the number is valid
if n < 1 or n > 5:
  print("Invalid number.")
else:
  # Get the file name corresponding to the number
  file = files[n-1]

  # Check if the file exists
  if os.path.isfile(file):
    # Copy the file and rename it to config.ini
    shutil.copy(file, "config.ini")
    print("File copied and renamed successfully.")
  else:
    print("File not found.")