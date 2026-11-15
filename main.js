"use strict";

const model = globalThis.BenchmarkModel;
const form = document.getElementById("loan-inputs");
const score = document.getElementById("score");
const detail = document.getElementById("score-detail");
const fill = document.getElementById("score-bar");
const numeric = key => form.elements.namedItem(key).valueAsNumber;
const percent = value => (value * 100).toFixed(1) + "%";
for (const [key,profile] of Object.entries(model.profiles)) {
  form.elements.history.add(new Option(profile.label,key));
}
form.elements.history.value = "regular";
for (const option of form.elements.history.options) option.defaultSelected = option.selected;
document.getElementById("auc").textContent = model.metrics.test_roc_auc.toFixed(3);
document.getElementById("test-size").textContent = model.metrics.test_rows.toLocaleString("en-US");
document.getElementById("prevalence").textContent = percent(model.metrics.test_prevalence);
document.getElementById("feature-count").textContent = model.metrics.feature_count;

function update() {
  if (!form.checkValidity()) {
    score.textContent = "Check inputs";
    fill.style.width = "0%";
    detail.textContent = "Enter a positive amount for each monetary field and an external score from 0 to 1, or leave that score blank.";
    return;
  }
  const external = numeric("external");
  const fields = {income:numeric("income"), credit:numeric("credit"), annuity:numeric("annuity"),
    goods:numeric("goods"), external:Number.isNaN(external) ? null : external};
  const history = form.elements.history.value;
  const value = CreditRisk.predict(model,CreditRisk.makeExample(model,history,fields));
  score.textContent = percent(value);
  fill.style.width = percent(value);
  detail.textContent = "Model score for the " + model.profiles[history].label.toLowerCase()
    + " example. The share labeled positive in the test set was " + percent(model.metrics.test_prevalence) + ".";
}
form.addEventListener("input",update);
form.addEventListener("change",update);
form.addEventListener("reset",() => queueMicrotask(update));
form.addEventListener("submit",event => {event.preventDefault();update();});
update();
