using System;
using System.IO;
using OPModel;
using OPModel.Types;

namespace BSTest
{
    class VanillaPayoffEvaluator : CMCEvaluator
    {
        public VanillaPayoffEvaluator(CModel model, double s)
        {
            numBatches = model.mcplan._nbatches;
            numScenPerBatch = model.mcplan._nscen_per_batch;
            lastStep = model.mcplan.nk - 1;
            gridMap = model.grid.host_d_xval_y;

            strike = s;

            results = new double[numBatches * numScenPerBatch];
        }

        public override unsafe void mc_eval_implementation(short* y_sk, int th, int batch)
        {
            short* y_s = &y_sk[lastStep * numScenPerBatch];

            fixed (double* results_p = &results[batch * numScenPerBatch])
            {
                for (int scen = 0; scen < numScenPerBatch; ++scen)
                {
                    short underlyingState = y_s[scen];
                    double underlyingVal = gridMap[underlyingState];

                    double payoff = Math.Max(underlyingVal - strike, 0);
                    results_p[scen] = payoff;
                }
            }
        }

        public double GetPayoff()
        {
            double total = 0;

            for (int batch = 0; batch < numBatches; ++batch)
            {
                for (int scen = 0; scen < numScenPerBatch; ++scen)
                {
                    total += results[batch * numScenPerBatch + scen];
                }
            }

            double payoff = total / (numBatches * numScenPerBatch);
            return payoff;
        }

        private int numScenPerBatch;
        private int lastStep;
        private double[] gridMap;
        private int numBatches;

        private double strike;

        private double[] results;
    }

    class OPPricer
    {
        public static double Price(double startVal, double strike, double riskFreeRate, double volatility, double yearsToMaturity,
            int numCurvePoints, int numSteps, double maxTimeStepDays, int numBatches, int numScenariosPerBatch, int numStates, int seed)
        {
            //Grid Setup
            int numPivotPoints = 7;
            double[] pivotPoints = new double[numPivotPoints];
            double[] gridSpacings = new double[numPivotPoints];
            pivotPoints[0] = 1; gridSpacings[0] = 5;
            pivotPoints[1] = 70; gridSpacings[1] = 2.5;
            pivotPoints[2] = 90; gridSpacings[2] = 1;
            pivotPoints[3] = 110; gridSpacings[3] = 2.5;
            pivotPoints[4] = 140; gridSpacings[4] = 5;
            pivotPoints[5] = 200; gridSpacings[5] = 7.5;
            pivotPoints[6] = 300; gridSpacings[6] = 10;

            //Setup interest rate curve and discount curve
            DateTime today = DateTime.Today;
            double daysInYear = 365.25;
            double daysToMaturity = yearsToMaturity * daysInYear;

            DateTime[] rateCurveTimes = BuildDates(today, daysToMaturity, numCurvePoints);

            double[] rates = new double[numCurvePoints];
            double[] discountFactors = new double[numCurvePoints];
            for (int iPoint = 0; iPoint < numCurvePoints; ++iPoint)
            {
                rates[iPoint] = riskFreeRate;

                if (iPoint == 0)
                    discountFactors[0] = Math.Exp(-rates[iPoint] * (rateCurveTimes[0] - today).Days / daysInYear);
                else
                    discountFactors[iPoint] = discountFactors[iPoint - 1] * Math.Exp(-rates[iPoint] * (rateCurveTimes[iPoint] - rateCurveTimes[iPoint - 1]).Days / daysInYear);
            }

            //Create grid
            
            //CDevice device = new CDevice(EFloatingPointPrecision.bit32, EFloatingPointUnit.device, 0);
            CDevice device = new CDevice(EFloatingPointPrecision.bit64, EFloatingPointUnit.host, 0);
            
            S1DGrid grid = new S1DGrid(device, numStates, today, rateCurveTimes, TimeSpan.FromDays(maxTimeStepDays), startVal,
                pivotPoints, gridSpacings);

            //Setup drift and vol arrays for geometric brownian motion
            double[][] drift = new double[numCurvePoints][];
            double[][] vol = new double[numCurvePoints][];
            for (int iCurvePoint = 0; iCurvePoint < numCurvePoints; ++iCurvePoint)
            {
                drift[iCurvePoint] = new double[numStates];
                vol[iCurvePoint] = new double[numStates];

                for (int iState = 0; iState < numStates; ++iState)
                {
                    drift[iCurvePoint][iState] = rates[iCurvePoint] * grid.host_d_xval(iState);
                    vol[iCurvePoint][iState] = volatility * grid.host_d_xval(iState);
                }
            }

            //Create model
            CLVModel model = new CLVModel(grid, "BSTestModel");
            model.mkgen(drift, vol); //drift and vol are expressed annually
            model.set_discount_curve(discountFactors);

            DateTime[] stepTimes = BuildDates(today, daysToMaturity, numSteps);

            model.make_mc_plan(numScenariosPerBatch, numBatches, stepTimes);
            model.exe_mc_plan();

            VanillaPayoffEvaluator evaluator = new VanillaPayoffEvaluator(model, strike);

            //model.device_mc_init();
            //model.device_mc_run1f(null, evaluator); //first parameter is not used

            if (seed != 0) model.host_d_mc_init_seed(seed);
            else model.host_d_mc_init();
            model.host_d_mc_run1f(null, evaluator); //first parameter is not used

            double payoff = evaluator.GetPayoff();

            //do discounting
            double pv = payoff * discountFactors[discountFactors.Length - 1];

            return pv;
        }

        //Construct a sequence of dates from start to start + days. Does not include start date in result.
        private static DateTime[] BuildDates(DateTime start, double days, int nSteps)
        {
            DateTime[] dates = new DateTime[nSteps];
            double daysPerStep = days / nSteps;
            for (int iStep = 0; iStep < nSteps - 1; ++iStep)
            {
                dates[iStep] = start.AddDays((int)((iStep + 1) * daysPerStep));
            }
            dates[nSteps - 1] = start.AddDays((int)days); //Ensure the end date is accurate

            return dates;
        }
    }
}
