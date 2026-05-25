import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
try: import seaborn as sns; HAS_SNS=True
except: HAS_SNS=False
from preprocess import get_train_datagen, get_val_test_datagen, IMG_SIZE

BATCH=32; EPOCHS=10; LR=1e-3
TRAIN_DIR="dataset/train"; VAL_DIR="dataset/val"; TEST_DIR="dataset/test"
MODEL_PATH="models/pneumonia_model.h5"
os.makedirs("models", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

def build_model():
    # No BatchNormalization — compatible with ALL TensorFlow 2.x versions
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(32,  (3,3), activation="relu", padding="same")(inp)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(64,  (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D(2,2)(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D(2,2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation="sigmoid")(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    m.summary()
    return m

def get_gens():
    td=get_train_datagen(); vd=get_val_test_datagen()
    tg=td.flow_from_directory(TRAIN_DIR,target_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH,class_mode="binary",shuffle=True)
    vg=vd.flow_from_directory(VAL_DIR,  target_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH,class_mode="binary",shuffle=False)
    eg=vd.flow_from_directory(TEST_DIR, target_size=(IMG_SIZE,IMG_SIZE),batch_size=BATCH,class_mode="binary",shuffle=False)
    return tg,vg,eg

if __name__=="__main__":
    tg,vg,eg = get_gens()
    print(f"Train:{tg.samples} | Val:{vg.samples} | Test:{eg.samples}")
    model = build_model()
    cbs = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1)
    ]
    h = model.fit(tg, epochs=EPOCHS, validation_data=vg, callbacks=cbs, verbose=1)
    print(f"\nTrain Acc: {h.history['accuracy'][-1]*100:.2f}%")
    print(f"Val   Acc: {h.history['val_accuracy'][-1]*100:.2f}%")
    loss,acc = model.evaluate(eg, verbose=1)
    print(f"Test  Acc: {acc*100:.2f}%")
    eg.reset()
    preds = (model.predict(eg, verbose=0)>0.5).astype(int).flatten()
    print(classification_report(eg.classes, preds, target_names=["NORMAL","PNEUMONIA"]))
    cm = confusion_matrix(eg.classes, preds)
    fig,ax = plt.subplots(figsize=(6,5))
    if HAS_SNS:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["NORMAL","PNEUMONIA"], yticklabels=["NORMAL","PNEUMONIA"])
    else:
        ax.imshow(cm, cmap="Blues")
        [ax.text(j,i,cm[i,j],ha="center",va="center") for i in range(2) for j in range(2)]
    ax.set_title("Confusion Matrix"); ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    plt.tight_layout(); plt.savefig("static/images/confusion_matrix.png"); plt.close()
    fig2,ax2=plt.subplots(1,2,figsize=(12,4))
    ax2[0].plot(h.history["accuracy"],label="Train"); ax2[0].plot(h.history["val_accuracy"],label="Val"); ax2[0].set_title("Accuracy"); ax2[0].legend()
    ax2[1].plot(h.history["loss"],label="Train");     ax2[1].plot(h.history["val_loss"],label="Val");     ax2[1].set_title("Loss");     ax2[1].legend()
    plt.tight_layout(); plt.savefig("static/images/training_history.png"); plt.close()
    print("[DONE] Training complete! Model saved to:", MODEL_PATH)
