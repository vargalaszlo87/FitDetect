

const FitDetect = {

    normalize (data) {
        const min = Math.min(...data);
        const max = Math.max(...data);
        return data.map(value => (value - min) / (max - min))
    },

    pearsonCorrelation (x, y) {
        const n = x.length;
        const meanX = x.reduce((a, b) => a + b, 0) / n;
        const meanY = y.reduce((a, b) => a + b, 0) / n;
        const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
        const denominator = Math.sqrt(
            x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0) *
            y.reduce((sum, yi) => sum + Math.pow(yi - meanY, 2), 0)
        );
        return numerator / denominator;
    },

    linearRegression (x, y) {
        const n = x.length;
        const meanX = x.reduce((a, b) => a + b, 0) / n;
        const meanY = y.reduce((a, b) => a + b, 0) / n;
        const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
        const denominator = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
        const slope = numerator / denominator; // m
        const intercept = meanY - slope * meanX; // c
        return { slope, intercept };
    },

    meanSquaredError (yTrue, yPred) {
        const n = yTrue.length;
        return yTrue.reduce((sum, yi, i) => sum + Math.pow(yi - yPred[i], 2), 0) / n;
    },

    identifyFunctionType(x, y) {
        // 1. Linear
        const linearModel = this.linearRegression(x, y);
        const linearPred = x.map(xi => linearModel.slope * xi + linearModel.intercept);
        const linearError = this.meanSquaredError(y, linearPred);
    
        // 2. Logarithmic
        const logX = x.map(xi => Math.log(xi));
        const logModel = this.linearRegression(logX, y);
        const logPred = x.map(xi => logModel.slope * Math.log(xi) + logModel.intercept);
        const logError = this.meanSquaredError(y, logPred);
    
        // 3. Power
        const logY = y.map(yi => Math.log(yi));
        const powerModel = this.linearRegression(logX, logY);
        const powerPred = x.map(xi => Math.exp(powerModel.intercept) * Math.pow(xi, powerModel.slope));
        const powerError = this.meanSquaredError(y, powerPred);
    
        // 4. Exponential
        const expLogY = y.map(yi => Math.log(yi));
        const expModel = this.linearRegression(x, expLogY);
        const expPred = x.map(xi => Math.exp(expModel.intercept) * Math.exp(expModel.slope * xi));
        const expError = this.meanSquaredError(y, expPred);
    
        // Legkisebb hiba alapján döntünk
        const errors = { linear: linearError, logarithmic: logError, power: powerError, exponential: expError };
        const bestFit = Object.keys(errors).reduce((a, b) => (errors[a] < errors[b] ? a : b));
    
        return { bestFit, errors };
    }

}

const x = [1, 2, 3, 4, 5];
const y = [2, 4, 6, 8, 10]; // Lineáris
console.log(FitDetect.identifyFunctionType(x, y));
