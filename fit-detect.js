/* NOTE FitDetect v.1.0.0
 *
 * fit-detect.js
 *
 * This is a function fitting application. The first method is an explicit 
 * regression method, the second is an implicit method with MPL (by tensor-flow)
 *
 * Copyright (C) 2024 Varga Laszlo
 * 
 * https://github.com/vargalaszlo87/FitDetect
 * http://vargalaszlo.com
 * http://ha1cx.hu
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.függvény illesztésre használható applikáció.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 2024-08-26
 */


const FitDetect = {

/* ANCHOR setup variable

    You can set the operations of fit detector in this section.

    typeOfCalcError:    set the type of error calculation
    alwaysNormalize:    you can set the normalize for all dataset
    epsilon:            if you have null in your dataset, then change it epsilon
    windowSizeOfAvg:    size of the window of the moving average filter

------------------------------------------------*/

    typeOfCalcError: "MSE",
    alwaysNormalize: false,
    epsilon: 1e-12,
    windowSizeOfAvg: 3,

/* ANCHOR math functions

    Here are the cores of the math functions.

    | Methods:

    normalize (data)
    pearsonCorrelation (x, y)
    linearRegression (x, y)
    quadraticRegression(x, y)
    sinusoidalRegression(x, y)
    sinusoidalRegression(x, y)
    exponentialDecayRegression(x, y)
    rationalRegression(x, y)

    | Fit test:

    Linear regression with transform of dataset
    Linear                          ( y = mx + c )  
    Logaritmic                      ( y = a + b * log(x) )      x --> log(x)
    Power                           ( y = a * x^b )             x,y --> log(x), log(y)
    Exponential                     ( y = a * e^bx )            y --> log(y)

    Quadratic regression            ( y = ax^2 + bx + c )
    Estimate sin                    ( y = a * sin(bx + c) + d )
    Rational regression             ( y = a / (x + b) + c )
    Polinomal regression            ( y = a_n * x^n + a_n-1 + x^n-1 + ... + a_1 * x + a_0 ) 
    Tanh regression                 ( y = a * tanh(b * x + c) + d )
    Sigmoid regression              ( y = 1 / ( 1 + e^-k*(x - x_0)) )

------------------------------------------------*/

    normalize (data) {
        const min = Math.min(...data);
        const max = Math.max(...data);
            return data.map(value => {
                let normalizedValue = (value - min) / (max - min);
                if (this.epsilon > 0 && normalizedValue === 0)
                    normalizedValue += this.epsilon;
                return normalizedValue;
            });
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

    rationalRegression(x, y) {
        const recipY = y.map(yi => 1 / yi);
        const model = this.linearRegression(x, recipY);
        const a = 1 / model.slope;
        const b = -model.intercept / model.slope;
        return { a, b };
    },  

    tanhRegression(x, y) {
        const meanY = y.reduce((a, b) => a + b, 0) / y.length;
        const amplitude = (Math.max(...y) - Math.min(...y)) / 2;
        const scale = 1; // Kezdeti tipp
        const offset = 0; // Kezdeti tipp
        return { amplitude, scale, offset, meanY };
    },

    sigmoidRegression(x, y) {
        const meanX = x.reduce((a, b) => a + b, 0) / x.length;
        const meanY = y.reduce((a, b) => a + b, 0) / y.length;
        const scale = 1 / (Math.max(...y) - Math.min(...y)); // Kezdeti tipp
        return { scale, inflectionPoint: meanX, offset: meanY };
    },
    
    
/* ANCHOR Error 

    This version calcualtes thre methods as:

        MSE - Mean Squared Error,
        RMSE - Root Mean Squared Error,
        MAE - Mean Absolute Error.

    you can choose by calcError function.

------------------------------------------------*/

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

/* ANCHOR filter function

    Filter function for dataset prefilter.

------------------------------------------------*/

    movingAverage(data, windowSize = this.windowSizeOfAvg) {
        return data.map((_, idx, arr) =>
            arr.slice(Math.max(0, idx - windowSize + 1), idx + 1)
            .reduce((sum, val) => sum + val, 0) / Math.min(idx + 1, windowSize)
        );
    },


/* ANCHOR main function

    This is the main method.  

------------------------------------------------*/

    identifyFunctionType(x, y) {

        // Prefilter
        /* TODO logic for prefilter */

        // Normalize
        if (this.alwaysNormalize) {
            x = this.normalize(x);
            y = this.normalize(y);
        }

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

        // Tanh
        const tanhModel = this.tanhRegression(x, y);
        const tanhPred =  x.map(xi =>
            tanhModel.amplitude * Math.tanh(tanhModel.scale * xi + tanhModel.offset) + tanhModel.meanY
        );
        const tanhError = this.meanSquaredError(y, tanhPred);

        // Sigmoid
        const sigmoidModel = this.sigmoidRegression(x, y);
        const sigmoidPred = x.map(xi =>
            1 / (1 + Math.exp(-sigmoidModel.scale * (xi - sigmoidModel.inflectionPoint))) + sigmoidModel.offset
        );
        const sigmoidError = this.meanSquaredError(y, sigmoidPred);
    
        // Legkisebb hiba alapján döntünk
        const errors = {
            linear: linearError,
            logarithmic: logError,
            power: powerError,
            exponential: expError,
            quadratic: quadraticError,
            sinusoidal: sinusoidalError,
            rational: rationalError,
            tanh: tanhError,
            sigmoid: sigmoidError
        };
        const bestFit = Object.keys(errors).reduce((a, b) => {
            if (isNaN(errors[a])) return b;
            if (isNaN(errors[b])) return a;
            return errors[a] < errors[b] ? a : b;
        });
    
        return { bestFit, errors };
    }

}

//const x = [1, 2, 3, 4, 5];
//const y = [1.98, 4, 6, 8, 10];

const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100];
const y = [1.0512710963760241, 1.1051709180756477, 1.161834242728283, 1.2214027581601699, 1.2840254166877414, 1.3498588075760032, 1.4190675485932573, 1.4918246976412703, 1.568312185490169, 1.6487212707001282, 1.7332530178673953, 1.8221188003905089, 1.9155408290138962, 2.0137527074704766, 2.117000016612675, 2.225540928492468, 2.3396468519259908, 2.45960311115695, 2.585709659315847, 2.718281828459045, 2.857651118063164, 3.0041660239464334, 3.158192909689768, 3.3201169227365472, 3.4903429574618414, 3.6692966676192444, 3.857425530696974, 4.055199966844675, 4.263114515168817, 4.4816890703380645, 4.711470182590742, 4.953032424395115, 5.206979827179849, 5.473947391727201, 5.754602676005731, 6.049647464412947, 6.359819522601832, 6.6858944422792685, 7.028687580589293, 7.38905609893065, 7.767901106306136, 8.166169912567652, 8.584858397177893, 9.025013499434122, 9.487735836358526, 9.974182454814724, 10.485569724727576, 11.023176380641605, 11.588346719223392, 12.182493960703473, 12.80710378266304, 13.463738035001692, 14.154038645375806, 14.879731724872837, 15.642631884188171, 16.444646771097045, 17.28778184056764, 18.174145369443067, 19.105953728231804, 20.085536923187668, 21.115344422540607, 22.197951281441636, 23.336064580942718, 24.53253019710935, 25.790339917193062, 27.112638920657887, 28.502733643767295, 29.964100047397012, 31.500392308747937, 33.11545195869231, 34.81331748760202, 36.59823444367799, 38.47466604903212, 40.4473043600674, 42.52108200006278, 44.701184493300835, 46.993063231579285, 49.40244910553017, 51.93536683483144, 54.598150033144236, 57.39745704544619, 60.340287597362, 63.43400029812333, 66.68633104092515, 70.10541234668786, 73.69979369959579, 77.47846292526087, 81.45086866496814, 85.62694400226295, 90.01713130052181, 94.63240831492406, 99.48431564193386, 104.58498557796919, 109.94717245212352, 115.58428452718766, 121.51041751873483, 127.74038984628504, 134.28977968493552, 141.17496392147686, 148.4131591025766];

console.log(x.length, y.length);


console.log(FitDetect.identifyFunctionType(x, y));
