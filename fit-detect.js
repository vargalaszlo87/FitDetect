
/*
    Linear regression with transform of dataset
    Linear          ( y = mx + c )  
    Logaritmic      ( y = a + b * log(x) )      x --> log(x)
    Power           ( y = a * x^b )             x,y --> log(x), log(y)
    Exponential     ( y = a * e^bx )            y --> log(y)

    Quadratic regression
                    ( y = ax^2 + bx + c )

    Estimate sin
                    ( y = a * sin(bx + c) + d )

    Exponential decay regression
                    ( y = a * e^-bx + c )

    Rational regression
                    ( y = a / (x + b) + c )

    Polinomal regression
                    ( y = a_n * x^n + a_n-1 + x^n-1 + ... + a_1 * x + a_0 ) 

*/

const FitDetect = {

    typeOfCalcError: "MSE",

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

    quadraticRegression(x, y) {
        const n = x.length;
        const xSquared = x.map(xi => xi * xi);
        const meanX = x.reduce((a, b) => a + b, 0) / n;
        const meanXSquared = xSquared.reduce((a, b) => a + b, 0) / n;
        const meanY = y.reduce((a, b) => a + b, 0) / n;
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2Y = xSquared.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const det = n * meanXSquared - meanX * meanX;
        const a = (n * sumX2Y - meanX * sumXY) / det;
        const b = (meanXSquared * sumXY - meanX * sumX2Y) / det; 
        const c = meanY - a * meanXSquared - b * meanX; 
    
        return { a, b, c };
    },

    sinusoidalRegression(x, y) {
        const meanY = y.reduce((a, b) => a + b, 0) / y.length;
        const amplitude = (Math.max(...y) - Math.min(...y)) / 2;
        const frequency = (2 * Math.PI) / (x[x.length - 1] - x[0]); // Alap frekvencia
        return { amplitude, frequency, phase: 0, offset: meanY };
    },

    exponentialDecayRegression(x, y) {
        const logY = y.map(yi => Math.log(yi)); // Logaritmus az y-ra
        const model = this.linearRegression(x, logY);
        const a = Math.exp(model.intercept); // Az e alapú exponenciális faktor
        const b = -model.slope; // Az exponenciális csökkenési sebesség
        return { a, b };
    },

    rationalRegression(x, y) {
        const recipY = y.map(yi => 1 / yi);
        const model = this.linearRegression(x, recipY);
        const a = 1 / model.slope;
        const b = -model.intercept / model.slope;
        return { a, b };
    },    

    meanSquaredError (yTrue, yPred) {
        if (yTrue.length != yPred.length)
            return;
        const n = yTrue.length;
        return yTrue.reduce((acc, yi, i) => acc + Math.pow(yi - yPred[i], 2), 0) / n;
    },

    rootMeanSquaredError (yTrue, yPred) {
        const mse = this.meanSquaredError(yTrue, yPred);
        return Math.sqrt(mse);
    },

    meanAbsoluteError(yTrue, yPred) {
        if (yTrue.length != yPred.length)
            return;
        const totalError = yTrue.reduce((acc, value, index) => {
            return acc + Math.abs(value - yPred[index]);
        }, 0);
        return totalError / yTrue.length;
    },

    calcError(yTrue, yPred, calcType = this.typeOfCalcError) {
        switch (calcType) {
            case "MSE": 
                return this.meanSquaredError(yTrue, yPred);
            case "RMSE":
                return this.rootMeanSquaredError(yTrue, yPred);
            case "MAE":
                return this.meanAbsoluteError(yTrue, yPred);
            default:
                throw new Error("Wrong calcType: "+ calcType);
        }
    },

    identifyFunctionType(x, y) {
        // Linear
        const linearModel = this.linearRegression(x, y);
        const linearPred = x.map(xi => linearModel.slope * xi + linearModel.intercept);
        const linearError = this.calcError(y, linearPred);
    
        // 2. Logarithmic
        const logX = x.map(xi => Math.log(xi));
        const logModel = this.linearRegression(logX, y);
        const logPred = x.map(xi => logModel.slope * Math.log(xi) + logModel.intercept);
        const logError = this.calcError(y, logPred);
    
        // Power
        const logY = y.map(yi => Math.log(yi));
        const powerModel = this.linearRegression(logX, logY);
        const powerPred = x.map(xi => Math.exp(powerModel.intercept) * Math.pow(xi, powerModel.slope));
        const powerError = this.calcError(y, powerPred);
    
        // Exponential
        const expLogY = y.map(yi => Math.log(yi));
        const expModel = this.linearRegression(x, expLogY);
        const expPred = x.map(xi => Math.exp(expModel.intercept) * Math.exp(expModel.slope * xi));
        const expError = this.calcError(y, expPred);

        // Exponential Decay
        const decayModel = this.exponentialDecayRegression(x, y);
        const decayPred = x.map(xi => decayModel.a * Math.exp(-decayModel.b * xi));
        const decayError = this.calcError(y, decayPred);

        // Quadratic
        const quadraticModel = this.quadraticRegression(x, y);
        const quadraticPred = x.map(xi => 
            quadraticModel.a * xi * xi + quadraticModel.b * xi + quadraticModel.c
        );
        const quadraticError = this.calcError(y, quadraticPred);

        // Sinusoidal
        const sinusoidalModel = this.sinusoidalRegression(x, y);
        const sinusoidalPred = x.map(xi =>
            sinusoidalModel.amplitude * Math.sin(sinusoidalModel.frequency * xi + sinusoidalModel.phase) +
            sinusoidalModel.offset
        );
        const sinusoidalError = this.calcError(y, sinusoidalPred);

        // Rational
        const rationalModel = this.rationalRegression(x, y);
        const rationalPred = x.map(xi => rationalModel.a / (xi + rationalModel.b));
        const rationalError = this.calcError(y, rationalPred);  
    
        // Legkisebb hiba alapján döntünk
        const errors = {
            linear: linearError,
            logarithmic: logError,
            power: powerError,
            exponential: expError,
            quadratic: quadraticError,
            sinusoidal: sinusoidalError,
            exponentialDecay: decayError,
            rational: rationalError
        };
        const bestFit = Object.keys(errors).reduce((a, b) => (errors[a] < errors[b] ? a : b));
    
        return { bestFit, errors };
    }

}

const x = [1, 2, 3, 4, 5];
const y = [2, 4, 6, 8, 10]; // Lineáris
console.log(FitDetect.identifyFunctionType(x, y));
