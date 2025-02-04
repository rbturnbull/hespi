const convertOutputPath = (filePath: string, outDirName: string = "hespi-output/") => {
  var cleanPath = filePath.split(outDirName)[1];
  var res = "/hespi-output/" + cleanPath;
  return res
};

const outDir = "/Users/gabrielem/GitHub/hespi-gui/"

module.exports = { convertOutputPath, outDir }