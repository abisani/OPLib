using System;
using System.IO;

namespace BSTest
{
    class Program
    {
        static void Main(string[] args)
        {
            //Option definition
            double strike = 100;
            double yearsToMaturity = 1;

            //Market Data
            double startVal = 100;
            double riskFreeRate = 0.05;

            //BS Model Parameter
            double volatility = 0.25;

            double bsOptionPrice = BSPricer.Price(startVal, strike, riskFreeRate, volatility, yearsToMaturity);

            System.Console.WriteLine("BS Option Price: " + bsOptionPrice);

            //Additional OP model parameters
            int numCurvePoints = 1;
            int numSteps = 1;
            double maxTimeStepDays = 1;
            int numBatches = 120;
            int numScenariosPerBatch = 4096 * 25;
            int numStates = 256;
            int seed = 0; //0 means no seed
           
            double opOptionPrice = OPPricer.Price(startVal, strike, riskFreeRate, volatility, yearsToMaturity,
                numCurvePoints, numSteps, maxTimeStepDays, numBatches, numScenariosPerBatch, numStates, seed);

            System.Console.WriteLine("OP Option Price: " + opOptionPrice);

            Console.Read();

        }
    }
}