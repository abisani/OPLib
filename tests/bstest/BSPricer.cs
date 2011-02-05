using System;

namespace BSTest
{
    class BSPricer
    {
        public static double Price(double s, double k, double r, double v, double t)
        {
            double d1 = (Math.Log(s / k) + (r + 0.5 * v * v) * t) / (v * Math.Sqrt(t));
            double d2 = d1 - v * Math.Sqrt(t);
            double price = s * N(d1) - k * Math.Exp(-r * t) * N(d2);
            return price;
        }

        private static double N(double x)
        {
            double y;
            double Exponential;
            double SumA;
            double SumB;
            double result;

            y = Math.Abs(x);
            if (y > 37)
            {
                result = 0;
            }
            else
            {
                Exponential = Math.Exp(-(Math.Pow(y, 2)) / 2);
                if (y < 7.07106781186547)
                {
                    SumA = 3.52624965998911E-02 * y + 0.700383064443688;
                    SumA = SumA * y + 6.37396220353165;
                    SumA = SumA * y + 33.912866078383;
                    SumA = SumA * y + 112.079291497871;
                    SumA = SumA * y + 221.213596169931;
                    SumA = SumA * y + 220.206867912376;
                    SumB = 8.83883476483184E-02 * y + 1.75566716318264;
                    SumB = SumB * y + 16.064177579207;
                    SumB = SumB * y + 86.7807322029461;
                    SumB = SumB * y + 296.564248779674;
                    SumB = SumB * y + 637.333633378831;
                    SumB = SumB * y + 793.826512519948;
                    SumB = SumB * y + 440.413735824752;
                    result = Exponential * SumA / SumB;
                }
                else
                {
                    SumA = y + 0.65;
                    SumA = y + 4 / SumA;
                    SumA = y + 3 / SumA;
                    SumA = y + 2 / SumA;
                    SumA = y + 1 / SumA;
                    result = Exponential / (SumA * 2.506628274631);
                }
            }

            if (x > 0)
            {
                result = 1.0 - result;
            }

            return result;
        }
    }
}
