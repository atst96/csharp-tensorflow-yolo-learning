using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlowSharpSSD
{
    internal static class Extensions
    {
        public static T GetValue<T>(this TFTensor tensor, bool jagged = true)
        {
            if (tensor.Shape.Length == 1 && tensor.Shape[0] == 0)
            {
                return default;
            }

            var value = tensor.GetValue(jagged);

            if (value == default)
            {
                return default;
            }

            return (T)value;
        }

        public static T GetValue<T>(this TFTensor[] tensor, int index, bool jagged = true)
        {
            return tensor[index].GetValue<T>(jagged);
        }
    }
}
