/* ============================================================
   CONFIGURATION
============================================================ */

// CloudFront distribution (NO trailing slash)
const WEBSITE_BASE_URL = "";

// Results live inside the same bucket
const RESULTS_BASE = `${WEBSITE_BASE_URL}/results`;

// Upload bucket
const USERDATA_BUCKET_URL =
  "https://ml-userdata-website.s3.ap-south-1.amazonaws.com";

const UPLOAD_PREFIX = "user-uploads";

const MODELS = [
  "logistic_regression",
  "random_forest",
  "xgboost",
  "neural_net"
];

const REFRESH_INTERVAL_MS = 300000; // 5 mins refresh


/* ============================================================
   DATASET UPLOAD
============================================================ */

async function uploadFile() {
  const input = document.getElementById("fileInput");
  const status = document.getElementById("uploadStatus");

  if (!input.files.length) {
    status.innerText = "Please select a CSV file.";
    return;
  }

  const file = input.files[0];
  const key = `${UPLOAD_PREFIX}/${Date.now()}-${file.name}`;
  const uploadUrl = `${USERDATA_BUCKET_URL}/${key}`;

  status.innerText = "Uploading dataset...";

  try {
    await fetch(uploadUrl, {
      method: "PUT",
      headers: { "Content-Type": "text/csv" },
      body: file
    });

    status.innerText =
      "Upload successful. Training pipeline triggered automatically.";

    input.value = "";
  } catch (err) {
    console.error(err);
    status.innerText = "Upload failed. Check console.";
  }
}


/* ============================================================
   MODEL RENDERING
============================================================ */

function modelImages(model) {
  switch (model) {
    case "logistic_regression":
      return ["accuracy.png", "confusion_matrix.png"];

    case "random_forest":
      return ["feature_importance.png"];

    case "xgboost":
      return ["training_curve.png"];

    case "neural_net":
      return ["roc_curve.png"];

    default:
      return [];
  }
}

async function loadModels() {
  const container = document.getElementById("model-cards");
  container.innerHTML = "";

  for (const model of MODELS) {
    try {
      const metricsUrl =
        `${RESULTS_BASE}/${model}/metrics.json?ts=${Date.now()}`;

      const metrics = await fetch(metricsUrl, { cache: "no-store" })
        .then(r => {
          if (!r.ok) throw new Error("metrics.json not found");
          return r.json();
        });

      const metricValue = Object.values(metrics).slice(-1)[0];

      const card = document.createElement("div");
      card.className = "model-card";

      const images = modelImages(model)
        .map(img =>
          `<img src="${RESULTS_BASE}/${model}/${img}?ts=${Date.now()}"
                onerror="this.style.display='none'">`
        )
        .join("");

      card.innerHTML = `
        <h3>${metrics.model_name}</h3>
        <p><strong>${metrics.metric_name}:</strong> ${metricValue}</p>

        <div class="plots">${images}</div>

        <div class="downloads">
          <a href="${RESULTS_BASE}/${model}/model.joblib" download>
            <button>Download Model</button>
          </a>
          <a href="${RESULTS_BASE}/${model}/weights.json" download>
            <button>Download Weights</button>
          </a>
          <a href="${RESULTS_BASE}/${model}/metrics.json" download>
            <button>Download Metrics</button>
          </a>
        </div>
      `;

      container.appendChild(card);

    } catch (err) {
      console.warn(`Skipping ${model}:`, err.message);
    }
  }
}


/* ============================================================
   INIT
============================================================ */

loadModels();
setInterval(loadModels, REFRESH_INTERVAL_MS);
