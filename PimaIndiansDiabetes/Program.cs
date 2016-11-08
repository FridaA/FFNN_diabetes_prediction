using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PimaIndiansDiabetes
{
    class Program
    {
        static void Main(string[] args)
        {
            PimaIndians diabetesPredictor = new PimaIndians("diabetes_data.txt", 0.7);
            diabetesPredictor.PredictDiabetes();
        }
    }
}
