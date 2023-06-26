using System.Drawing;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceAuth;

public class FaceAuthProvider
{
    #region Vaiable

    private Net net;
    private FaceRecognizer Recognizer;
    private FaceRecognizer.PredictionResult result = new();


    private List<Image<Gray, byte>> imgList = new();
    private VideoCapture capture;

    private const string dbFolderName = "TrainedFaces";

    private int[] FaceLabels;
    private int FaceThreshold = 3500;

    const int Width = 200;
    const int Height = 200;

    private float FaceDistance = -1;
    private bool _isTrained;

    string config = Directory.GetCurrentDirectory() + "\\Assets\\deploy.prototxt";
    string model = Directory.GetCurrentDirectory() + "\\Assets\\res10_300x300_ssd_iter_140000_fp16.caffemodel";

    #endregion


    public FaceAuthProvider()
    {
        net = DnnInvoke.ReadNetFromCaffe(config, model);
        Recognizer = new FisherFaceRecognizer(0, FaceThreshold);
    }

    public bool CameraInitialize()
    {
        if (!capture.IsOpened)
        {
            using (capture = new(0))
            {
                return false;
            }
        }

        return true;
    }

    public Mat? FaceDetect()
    {
        using var frame = new Mat();

        capture.Read(frame);

        if (frame.IsEmpty)
            throw new Exception("Received image is empty");

        // Creating a blob object from a frame to enter the neural network
        var blob = DnnInvoke.BlobFromImage(frame, 1.0, new Size(250, 250),
            new MCvScalar(104, 117, 123), false, false);

        net?.SetInput(blob);


        // Getting face detections
        var detections = net?.Forward("detection_out");

        if (detections != null)
        {
            int[] dim = detections.SizeOfDimension;
            float[,,,] values = detections.GetData(true) as float[,,,];
            for (int i = 0; i < dim[2]; i++)
            {
                // Получение координат и вероятности лица
                float confidence = values[0, 0, i, 2];
                int x1 = (int)(values[0, 0, i, 3] * frame.Width);
                int y1 = (int)(values[0, 0, i, 4] * frame.Height);

                // Отрисовка прямоугольника вокруг лица на изображении
                if (confidence > 0.5) // Фильтрация по порогу вероятности
                {
                    return new Mat(frame, new Rectangle(x1, y1, Width, Height)).Clone();
                }
            }
        }

        return null;
    }

    public bool Recognize(Mat face, int FaceThresh = -1)
    {
        if (_isTrained)
        {
            result = Recognizer.Predict(face.ToImage<Gray, byte>());

            if (result.Label != -1 && result.Label >= FaceLabels[0] && result.Label <= FaceLabels[^1])
            {
                FaceDistance = (float)result.Distance;

                if (FaceThresh > -1) FaceThreshold = FaceThresh;


                if (FaceDistance < FaceThreshold) return true;

                else return false;
            }
            else
            {
                return false;
            }
        }
        else
        {
            throw new Exception("Recognition Model not trained");
        }
    }


    public void RegisterIFace(string name, Mat face)
    {
        Random rand = new Random();
        string filename = "face_" + name + "_" + rand.Next() + ".jpg";
        const string folder = "TrainedFaces";
        if (!Directory.Exists(folder))
        {
            Directory.CreateDirectory(folder);
        }

        filename = Path.Combine(folder, filename);

        face.ToImage<Gray, byte>().Save(filename);
    }

    public void LoadFaceRecognizer()
    {
        if (!Directory.Exists(dbFolderName))
        {
            Directory.CreateDirectory(dbFolderName);
            Directory.CreateDirectory(AppDomain.CurrentDomain.BaseDirectory + "\\" + dbFolderName);
        }
        else
        {
            if (!_isTrained)
            {
                string[] files = Directory.GetFiles(dbFolderName, "*.jpg", SearchOption.AllDirectories);

                if (files.Length != 0)
                {
                    foreach (var file in files)
                    {
                        imgList.Add(new Image<Gray, byte>(file));
                    }


                    using VectorOfMat vectorOfMat = new VectorOfMat();
                    using VectorOfInt vectorOfInt =
                        new VectorOfInt(FaceLabels = Enumerable.Range(0, files.Length).ToArray());
                    vectorOfMat.Push(imgList.ToArray());

                    try
                    {
                        Recognizer.Train(vectorOfMat, vectorOfInt);
                        _isTrained = true;
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                        throw;
                    }
                }
            }
        }
    }
}