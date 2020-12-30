window.toggleScheme = function() {
  const body = document.querySelector("body")
  const preferSupported = window.matchMedia("(prefers-color-scheme)").media !== "not all"
  let scheme = body.getAttribute("data-md-color-scheme")
  let prefer = body.getAttribute("data-md-prefers-color-scheme")
  console.log(scheme, prefer)
  //if (preferSupported && scheme === "default" && prefer !== "true") {
    //prefer = "true"
    //scheme = (window.matchMedia("(prefers-color-scheme: dark)").matches) ? "slate" : "default"
    //console.log("1")
  //} else if (preferSupported && prefer === "true") {
    //prefer = "false"
    //scheme = "slate"
    //console.log("2")
  //} else if (scheme === "slate") {
  if (scheme === "iceberg") {
    prefer = "false"
    scheme = "default"
    console.log("3")
  } else {
    prefer = "false"
    scheme = "iceberg"
    console.log("4")
  }
  localStorage.setItem("data-md-prefers-color-scheme", prefer)
  body.setAttribute("data-md-prefers-color-scheme", prefer)
  body.setAttribute("data-md-color-scheme", scheme)
}

