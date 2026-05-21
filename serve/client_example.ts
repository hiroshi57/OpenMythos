/**
 * OpenMythos API — Node.js / TypeScript client example
 *
 * npm install node-fetch   (or use built-in fetch in Node 18+)
 */

const API_BASE = process.env.OPENMYTHOS_API ?? "http://localhost:8000";

interface InferRequest {
  text: string;
  loops?: number;
  task?: "identity_verify" | "fraud_detect" | "general";
}

interface InferResponse {
  score: number;
  label: number;  // 1 = verified / positive, 0 = flagged / negative
  loops_used: number;
  latency_ms: number;
  model_params: number;
}

async function infer(req: InferRequest): Promise<InferResponse> {
  const res = await fetch(`${API_BASE}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await res.text()}`);
  return res.json() as Promise<InferResponse>;
}

// --- Example 1: real-time identity verification (low loops = fast) ---
async function identityVerify(userText: string) {
  const result = await infer({
    text: userText,
    task: "identity_verify",   // server uses loops=4 automatically
  });
  console.log("[identity_verify]", result);
  return result.label === 1;
}

// --- Example 2: fraud detection (high loops = deep reasoning) ---
async function fraudDetect(transactionText: string) {
  const result = await infer({
    text: transactionText,
    task: "fraud_detect",      // server uses loops=12 automatically
  });
  console.log("[fraud_detect]", result);
  return result.label === 0;   // label 0 = flagged as fraud
}

// --- Example 3: explicit loop control ---
async function customInfer(text: string, loops: number) {
  return infer({ text, loops, task: "general" });
}

// --- Run examples ---
(async () => {
  const health = await fetch(`${API_BASE}/health`).then(r => r.json());
  console.log("[health]", health);

  await identityVerify("User login: takizawa.hiroshi, device fingerprint XYZ");
  await fraudDetect("Transfer $50,000 to account 9999 at 3am from new device");
  const custom = await customInfer("Sample text for general inference", 8);
  console.log("[custom loops=8]", custom);
})();
