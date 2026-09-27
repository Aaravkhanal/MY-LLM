/* Runs before the first paint (a classic script in <head>) so the chosen theme never flashes. Dark is the default. */
(function () {
  var t = "dark";
  try { var s = localStorage.getItem("aarav_theme"); if (s === "light" || s === "dark") t = s; } catch (e) { /* private mode: default */ }
  document.documentElement.setAttribute("data-theme", t);
})();
