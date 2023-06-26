using System.Drawing;
using System.Reflection;
using System.Xml.Linq;
using System.Xml.XPath;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceAuth;

/// <summary>
/// A class that generates methods for distinct and identifying faces using a neural network and the FisherFaceRecognizer algorithm.
/// </summary>
public class FaceAuthProvider
{
    #region Vaiable

    private Net net;
    private FaceRecognizer Recognizer;
    private FaceRecognizer.PredictionResult result;


    private List<Image<Gray, byte>> imgList;
    private VideoCapture capture;

    private int[] FaceLabels;
    private int FaceThreshold = 3500;

    public readonly int Width = 196;
    public readonly int Height = 257;

    private float FaceDistance = -1;
    private bool _isTrained;

    private const string DbFolderName = "TrainedFaces";
    
    private string config = Directory.GetCurrentDirectory() + "\\Assets\\deploy.prototxt";
    private string model = Directory.GetCurrentDirectory() + "\\Assets\\res10_300x300_ssd_iter_140000_fp16.caffemodel";

    #endregion
    

    /// <summary>
    /// The class constructor that initializes the neural network and the face recognizer.
    /// </summary>
    public FaceAuthProvider()
    {
        net = DnnInvoke.ReadNetFromCaffe(config, model);
        Recognizer = new FisherFaceRecognizer(0, FaceThreshold);
        imgList = new List<Image<Gray, byte>>();
    }

    /// <summary>
    /// Method that initializes capturing video from the camera.
    /// </summary>
    /// <returns>Returns true if the camera was opened successfully, false otherwise.</returns>
    public bool CameraInitialize()
    {
        capture = new VideoCapture(0);

        if (!capture.IsOpened)
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// A method that detects a face in a camera image using a neural network.
    /// </summary>
    /// <returns>Returns a Mat containing the image of the face, or null if no face was found.</returns>
    /// <exception cref="Exception">Throws an exception if the camera image is empty.</exception>
    public Mat? FaceDetect()
    {
        using var frame = new Mat();

        capture.Read(frame);

        if (frame.IsEmpty)
            throw new Exception("Received image is empty");

        // Creating a blob object from a frame to enter the neural network
        var blob = DnnInvoke.BlobFromImage(frame, 1.0, new Size(250, 250),
            new MCvScalar(104, 117, 123), false, false);

        net.SetInput(blob);


        // Getting face detections
        var detections = net.Forward("detection_out");

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

    /// <summary>
    /// A method that recognizes a face using the FisherFaceRecognizer algorithm.
    /// </summary>
    /// <returns>Returns true if the face is recognized, false otherwise.</returns>
    /// <exception cref="Exception">Throws an exception if the recognition model is not trained.</exception>
    public bool Recognize()
    {
        if (_isTrained)
        {
            Mat? face = FaceDetect();
            if (face != null)
            {
                result = Recognizer.Predict(face.ToImage<Gray, byte>());

                if (result.Label != -1 && result.Label >= FaceLabels[0] && result.Label <= FaceLabels[^1])
                {
                    FaceDistance = (float)result.Distance;


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
                return false;
            }
        }
        else
        {
            throw new Exception("Recognition Model not trained");
        }
    }

    /// <summary>
    /// A method that registers a face and saves its image in the TrainedFaces folder.
    /// </summary>
    /// <param name="name">Name of the person.</param>
    /// <param name="face">Image of a face type Mat.</param>
    public void RegisterIFace(string name, int count)
    {
        string NameFolder = $"{DbFolderName}\\{name}";
        for (int i = 0; i < count; i++)
        {
            Random rand = new Random();
            string filename = "face_" + name + "_" + rand.Next() + ".jpg";

            if (!Directory.Exists(NameFolder))
            {
                Directory.CreateDirectory(NameFolder);
            }

            filename = Path.Combine(NameFolder, filename);

            FaceDetect()?.ToImage<Gray, byte>().Save(filename);
        }
    }

    /// <summary>
    /// A method that loads a trained face recognizer model from the TrainedFaces folder.
    /// </summary>
    /// <exception cref="Exception">Throws an exception if model training fails.</exception>
    public void LoadFaceRecognizer()
    {
        if (!Directory.Exists(DbFolderName))
        {
            Directory.CreateDirectory(DbFolderName);
            Directory.CreateDirectory(AppDomain.CurrentDomain.BaseDirectory + "\\" + DbFolderName);
        }
        else
        {
            if (!_isTrained)
            {
                string[] files = Directory.GetFiles(DbFolderName, "*.jpg", SearchOption.AllDirectories);

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