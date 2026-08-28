// ----------------------------------------------------------------------
// Shared submit handler for the login and register pages.
// The server sets an HttpOnly session cookie on success — this script
// never touches or stores a token itself, it just redirects on success.
// ----------------------------------------------------------------------

function setupAuthForm({ formId, submitBtnId, submitLabel, loadingLabel, endpoint, buildPayload, validate }) {
  const form = document.getElementById(formId);
  const submitBtn = document.getElementById(submitBtnId);
  const errorBox = document.getElementById("auth-error");

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add("visible");
  }

  function hideError() {
    errorBox.textContent = "";
    errorBox.classList.remove("visible");
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    if (typeof validate === "function") {
      const validationError = validate();
      if (validationError) {
        showError(validationError);
        return;
      }
    }

    submitBtn.disabled = true;
    submitBtn.textContent = loadingLabel;

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(buildPayload()),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(extractErrorMessage(data));
      }

      window.location.href = "/";
    } catch (err) {
      showError(err.message);
      submitBtn.disabled = false;
      submitBtn.textContent = submitLabel;
    }
  });
}

function extractErrorMessage(data) {
  // FastAPI validation errors (422) return detail as a list of objects
  // rather than a plain string — surface the first one legibly.
  if (Array.isArray(data.detail)) {
    const first = data.detail[0];
    return (first && (first.msg || first.message)) || "Please check your input and try again.";
  }
  return data.detail || data.error || "Something went wrong. Please try again.";
}
