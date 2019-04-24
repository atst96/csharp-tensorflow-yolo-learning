using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace TensorFlowSharpSSD
{
    internal class ObjectDetector : IDisposable
    {
        private readonly TFSession _session;
        private readonly TFOutput _detectionInputTensor;
        private readonly TFOutput[] _detectionOutputTensors;

        public bool IsGpuMode { get; }
        public int Width { get; }
        public int Height { get; }

        public ObjectDetector(string graphPath, int width, int height, bool isGpuMode = false)
        {
            this.Width = width;
            this.Height = height;
            this.IsGpuMode = isGpuMode;

            using (var graph = new TFGraph())
            {
                var model = File.ReadAllBytes(graphPath);
                graph.Import(new TFBuffer(model));

                this._session = new TFSession(graph);
                this._detectionInputTensor = graph["Placeholder"][0];

                if (isGpuMode)
                {
                    this._detectionOutputTensors = new[]
                    {
                        graph["concat_10"][0],
                        graph["concat_11"][0],
                        graph["concat_12"][0],
                    };
                }
                else
                {
                    this._detectionOutputTensors = new[]
                    {
                        graph["concat_9"][0],
                        graph["mul_6"][0],
                    };
                }
            }
        }

        private static (TFSession session, TFOutput input, TFOutput output) CreateRawImageGraph(int imageWidth, int imageHeight)
        {
            var graph = new TFGraph();
            var input = graph.Placeholder(TFDataType.String);

            var maxValue = graph.Const(255.0f, TFDataType.Float);

            var shape = graph.Stack(new[]
            {
                graph.Const(-1, TFDataType.Int32),
                graph.Const(imageHeight, TFDataType.Int32),
                graph.Const(imageWidth, TFDataType.Int32),
                graph.Const(3, TFDataType.Int32),
            });

            TFOutput output;

            output = graph.DecodeRaw(input, TFDataType.UInt8);
            output = graph.Cast(output, TFDataType.Float);
            output = graph.Div(output, maxValue);
            output = graph.Reshape(output, shape);

            return (new TFSession(graph), input, output);
        }

        private TFTensor CreateTensor(byte[] imageData, int width, int height)
        {
            var (session, input, output) = CreateRawImageGraph(width, height);

            using (session)
            {
                var value = TFTensor.CreateString(imageData);

                return session.Run(
                    inputValues: new[] { value },
                    inputs: new[] { input },
                    outputs: new[] { output }
                )[0];
            }
        }

        public Box[][] Predict(byte[] data, int originalWidth, int originalHeight)
        {
            (int width, int height) = (this.Width, this.Height);

            var tensor = this.CreateTensor(data, width, height);

            var runner = this._session.GetRunner();

            var output = runner
                .AddInput(this._detectionInputTensor, tensor)
                .Fetch(this._detectionOutputTensors)
                .Run();

            float widthCoe = originalWidth / (float)this.Width;
            float heightCoe = originalHeight / (float)this.Height;

            if (this.IsGpuMode)
            {
                var boxes = output[0].GetValue<float[][]>();
                var scores = output[1].GetValue<float[]>();
                var labels = output[2].GetValue<int[]>();

                Box[] result;

                if (boxes == null || scores == null || labels == null)
                {
                    result = new Box[0];
                }
                else
                {
                    result = GetBoxes(width, height, widthCoe, heightCoe, boxes, scores, labels);
                }

                return new[] { result };
            }
            else
            {
                var boxes = output[0].GetValue<float[][][]>();
                var scores = output[1].GetValue<float[][][]>();

                throw new NotImplementedException();
            }

            //var boxes = output[0].GetValue<float[][][]>();
            //var scores = output[1].GetValue<float[][]>();
            //var classes = output[2].GetValue<float[][]>();

            //var predicts = new Box[boxes.Length][];

            //for (int i = 0; i < predicts.Length; ++i)
            //{
            //    predicts[i] = GetBoxes(width, height, boxes[i], scores[i], classes[i]);
            //}

            //return predicts;
        }

        public Box[] Predict(Bitmap image)
        {
            var data = ToBytes(image, this.Width, this.Height);

            return Predict(data, image.Width, image.Height)[0];
        }

        private static Box[] GetBoxes(int width, int height, float widthCoe, float heightCoe, float[][] boxes, float[] scores, int[] classes)
        {
            var results = new Box[boxes.Length];

            for (int i = 0; i < results.Length; ++i)
            {
                var box = boxes[i];
                (float x, float y) = (box[0] * widthCoe, box[1] * heightCoe);
                (float r, float b) = (box[2] * widthCoe, box[3] * heightCoe);

                int classId = classes[i] + 1;
                float score = scores[i];

                results[i] = new Box(classId, score, (int)x, (int)y, (int)(r - x), (int)(b - y));
            }

            return results;
        }

        private static byte[] ToBytes(Bitmap source, int width, int height)
        {
            Bitmap image = ResizeImage(source, width, height);

            var rect = new Rectangle(0, 0, width, height);
            var imageData = image.LockBits(rect, ImageLockMode.ReadOnly, image.PixelFormat);

            try
            {
                switch (image.PixelFormat)
                {
                    case PixelFormat.Format24bppRgb:
                        return ToRGBFrom24Bpp(imageData);

                    case PixelFormat.Format32bppArgb:
                        return ToRGBFrom32Bpp(imageData);

                    default:
                        throw new NotImplementedException();
                }
            }
            finally
            {
                image.UnlockBits(imageData);
                image.Dispose();
            }
        }

        private static Bitmap ResizeImage(Bitmap source, int width, int height)
        {
            var destImage = new Bitmap(width, height, source.PixelFormat);

            using (var g = Graphics.FromImage(destImage))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBilinear;
                g.DrawImage(source, 0, 0, width, height);
            }

            return destImage;
        }

        private static unsafe byte[] ToRGBFrom24Bpp(BitmapData imageData)
        {
            int width = imageData.Width;
            int height = imageData.Height;
            int destStride = width * sizeof(RGB);

            var dist = new byte[destStride * imageData.Height];

            RGB* srcImgPtr = (RGB*)imageData.Scan0;
            fixed (byte* destImgDataPinnedPtr = dist)
            {
                byte* destImageDataPtr = destImgDataPinnedPtr;

                Parallel.For(0, imageData.Height, y =>
                {
                    RGB* yDestRgbPtr = (RGB*)(destImageDataPtr + (destStride * y));
                    RGB* ySrcRgbPtr = srcImgPtr + (width * y);

                    for (int x = 0; x < width; ++x)
                    {
                        ySrcRgbPtr++->ReverseTo(yDestRgbPtr++);
                    }
                });
            }

            return dist;
        }

        private static unsafe byte[] ToRGBFrom32Bpp(BitmapData imageData)
        {
            int width = imageData.Width;
            int height = imageData.Height;
            int destStride = width * sizeof(RGB);

            var dist = new byte[imageData.Width * imageData.Height * 3];

            ARGB* srcImgPtr = (ARGB*)imageData.Scan0;
            fixed (byte* destImgDataPinnedPtr = dist)
            {
                byte* destImageDataPtr = destImgDataPinnedPtr;

                Parallel.For(0, imageData.Height, y =>
                {
                    RGB* yDestRgbPtr = (RGB*)(destImageDataPtr + (destStride * y));
                    ARGB* ySrcRgbPtr = srcImgPtr + (width * y);

                    for (int x = 0; x < width; ++x)
                    {
                        ySrcRgbPtr++->ReverseTo(yDestRgbPtr++);
                    }
                });
            }

            return dist;
        }

        public void Dispose()
        {
            this._session.Dispose();
        }
    }
}
