import os
import zipfile

# ==============================
# STEP 1: CREATE requirements.txt
# ==============================
requirements_content = """streamlit
numpy
scikit-learn
joblib
tensorflow
xgboost
"""

with open("requirements.txt", "w") as f:
    f.write(requirements_content)

print("✅ requirements.txt created")

# ==============================
# STEP 2: LIST FILES TO ZIP
# ==============================
files_to_zip = [
    "app.py",
    "requirements.txt"
]

# ==============================
# STEP 3: CREATE ZIP FILE
# ==============================
zip_name = "flight_app_project.zip"

with zipfile.ZipFile(zip_name, 'w') as zipf:
    for file in files_to_zip:
        if os.path.exists(file):
            zipf.write(file)
            print(f"✔ Added: {file}")
        else:
            print(f"❌ Missing: {file}")

print("\n🎉 ZIP file created:", zip_name)

# ==============================
# STEP 4: DOWNLOAD ZIP
# ==============================
from google.colab import files
files.download(zip_name)
