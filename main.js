const createRequestBody = (observationRows) => {
  const dataRow = observationRows.querySelector("tr.table-data-row");
  const [sourceName, observationID] = observationRows.id.split("/");
  const instrument = dataRow.firstElementChild.textContent;
  const newBinsize = observationRows.querySelector(
    "input.new-binsize-field"
  ).value;
  return { sourceName, observationID, instrument, newBinsize };
};

const modifyObservationRows = (observationRows, responseData) => {
  const dataRow = observationRows.querySelector("tr.table-data-row");
  const dataCells = dataRow.getElementsByTagName("td");
  const lightcurvePlotImage = observationRows.querySelector(
    "img.lightcurve-plot-image"
  );
  for (const i in dataCells) {
    dataCells[i].textContent = responseData.newData[i];
  }
  lightcurvePlotImage.src = responseData.newPlotPath;
};

const updateStatusCell = (observationRows, message) => {
  const statusCell = observationRows.querySelector("td.status-display");
  statusCell.textContent = message;
};

const handleResponse = (response) => {
  if (!response.ok) {
    if (response.status === 400) {
      throw new Error("Invalid binsize inputted."); // TODO display this in the status
    }
    throw new Error("Server error.");
  }
  return response.json();
};

document.addEventListener("click", (event) => {
  if (event.target.classList.contains("binsize-recalculation-button")) {
    const observationRows = event.target.closest("tbody");
    const eventSource = new EventSource("/rebinning_status");
    eventSource.onmessage = (event) => {
      updateStatusCell(observationRows, event.data);
    };
    console.log("Making request to server...");
    fetch("/recalculate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(createRequestBody(observationRows)),
    })
      .then((response) => handleResponse(response))
      .then((newData) => modifyObservationRows(observationRows, newData))
      .catch((error) => {
        console.error("There was an error making the request:", error.message);
      })
      .finally(() => console.log("Request complete."));
  }
});
