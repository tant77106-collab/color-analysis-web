from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


# -----------------------------
# หา K อัตโนมัติจากรูป
# -----------------------------
def find_best_k(pixels, max_k=12):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    last_compact = None
    best_k = 2

    for k in range(2, max_k + 1):
        compact, _, _ = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        if last_compact is not None:
            improvement = (last_compact - compact) / last_compact
            if improvement < 0.12:
                break

        last_compact = compact
        best_k = k

    return best_k


# -----------------------------
# วิเคราะห์สี (ขาว = พิมพ์)
# -----------------------------
def extract_colors(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (250, 250))

    pixels = img.reshape((-1, 3)).astype(np.float32)
    total_pixels = pixels.shape[0]

    K = find_best_k(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())

    colors = []

    # คำนวณพื้นที่สี
    for i in range(K):
        r, g, b = centers[i].astype(int)
        area = counts[i] / total_pixels * 100

        colors.append({
            "rgb": (r, g, b),
            "hex": rgb_to_hex((r, g, b)),
            "area": area,
            "printed": area   # ✅ ขาวก็นับพิมพ์
        })

    # -----------------------------
    # บังคับให้รวม = 100%
    # -----------------------------
    area_sum = sum(c["area"] for c in colors)
    diff = 100 - area_sum

    colors[0]["area"] += diff
    colors[0]["printed"] += diff

    # ปัดเลข
    for c in colors:
        c["area"] = round(c["area"], 2)
        c["printed"] = round(c["printed"], 2)

    printed_total = round(sum(c["printed"] for c in colors), 2)

    # เรียงจากมากไปน้อย
    colors.sort(key=lambda x: x["area"], reverse=True)

    return colors, printed_total


# -----------------------------
# Flask
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    colors = []
    image_url = None
    printed_total = 0

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            colors, printed_total = extract_colors(path)
            image_url = path

    return render_template(
        "index.html",
        colors=colors,
        image=image_url,
        printed_total=printed_total
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
