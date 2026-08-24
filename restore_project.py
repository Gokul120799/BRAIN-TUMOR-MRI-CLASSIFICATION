
import os
import shutil

# ============================================================
# BRAIN TUMOR PROJECT RESTORE SCRIPT
# ============================================================

PROJECT_DIR = "/content/drive/MyDrive/Brain_Tumor_Project"

FILES = [
    "app.py",
    "cnn_model.h5",
    "resnet_model.h5",
    "best_model.h5",
    "class_indices.pkl"
]

print("=" * 60)
print("🔄 RESTORING BRAIN TUMOR PROJECT")
print("=" * 60)

for filename in FILES:

    source = os.path.join(
        PROJECT_DIR,
        filename
    )

    destination = os.path.join(
        "/content",
        filename
    )

    if os.path.exists(source):

        shutil.copy2(
            source,
            destination
        )

        print(
            f"✅ Restored: {filename}"
        )

    else:

        print(
            f"❌ Missing from Drive: {filename}"
        )

print("=" * 60)
print("✅ RESTORE COMPLETE")
print("=" * 60)
