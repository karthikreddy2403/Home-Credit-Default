const assert = require("node:assert/strict");
const examples = require("./reference.json");
const { predict, makeExample } = require("../docs/predict.js");
require("../docs/model-data.js");
const model = globalThis.BenchmarkModel;
for (const row of examples) {
  assert.ok(Math.abs(predict(model, row.values) - row.score) < 1e-9, "LightGBM score mismatch");
}
const fields = { income: 150000, credit: 450000, annuity: 22500, goods: 400000, external: null };
for (const key of Object.keys(model.profiles)) {
  const input = makeExample(model, key, fields);
  assert.equal(input[model.feature_names.indexOf("CREDIT_TO_INCOME")], 3);
  assert.equal(input[model.feature_names.indexOf("EXT_SOURCE_2")], null);
  assert.ok(Number.isFinite(predict(model, input)));
}
console.log(examples.length + " Python/JavaScript parity checks passed.");
