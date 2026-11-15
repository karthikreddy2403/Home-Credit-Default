"use strict";

const CreditRisk = (() => {
  function leafValue(node, values) {
    while (Array.isArray(node)) {
      const [index, threshold, defaultLeft, missingType, left, right] = node;
      const raw = values[index];
      const missing = raw == null || Number.isNaN(raw);
      const value = missing ? 0 : Math.fround(raw);
      const useDefault = (missingType === "NaN" && missing)
        || (missingType === "Zero" && Math.abs(value) < 1e-35);
      node = (useDefault ? defaultLeft : value <= threshold) ? left : right;
    }
    return node;
  }

  function predict(model, values) {
    let logOdds = 0;
    for (const tree of model.trees) logOdds += leafValue(tree, values);
    return 1 / (1 + Math.exp(-logOdds));
  }

  function makeExample(model, profile, fields) {
    const features = model.profiles[profile].features.slice();
    const inputs = {
      AMT_INCOME_TOTAL: fields.income,
      AMT_CREDIT: fields.credit,
      AMT_ANNUITY: fields.annuity,
      AMT_GOODS_PRICE: fields.goods,
      EXT_SOURCE_2: fields.external,
      CREDIT_TO_INCOME: fields.credit / fields.income,
      ANNUITY_TO_INCOME: fields.annuity / fields.income,
      CREDIT_TO_GOODS: fields.credit / fields.goods,
      ANNUITY_TO_CREDIT: fields.annuity / fields.credit
    };
    for (const [key, value] of Object.entries(inputs)) {
      features[model.feature_names.indexOf(key)] = value;
    }
    return features;
  }
  return { predict, makeExample };
})();

if (typeof module !== "undefined") module.exports = CreditRisk;
