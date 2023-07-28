using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;
using FaceONNX;

namespace FaceAuth;

/// <summary>
/// A class that generates methods for distinct and identifying faces using a neural network and algorithm for linear discriminant analysis by Fisher's criterion.
/// </summary>
public class FaceAuthProvider
{
    #region Vaiable

    static FaceDetector faceDetector;
    static FaceLandmarksExtractor _faceLandmarksExtractor;
    static FaceEmbedder _faceEmbedder;
    private Embeddings embeddings;
    private VideoCapture capture;

    private const string DbFolderName = "RegisteredFaces";

    #endregion


    /// <summary>
    /// The class constructor that initializes the neural network, face recognizer and creating assets files..
    /// </summary>
    public FaceAuthProvider()
    {
        faceDetector = new FaceDetector();
        _faceLandmarksExtractor = new FaceLandmarksExtractor();
        _faceEmbedder = new FaceEmbedder();
        embeddings = new Embeddings();
    }

    static float[] GetEmbedding(Bitmap image)
    {
        var rectangles = faceDetector.Forward(image);
        var rectangle = rectangles.FirstOrDefault();

        if (!rectangle.IsEmpty)
        {
            // landmarks
            var points = _faceLandmarksExtractor.Forward(image, rectangle);
            var angle = points.GetRotationAngle();

            // alignment
            using var aligned = FaceLandmarksExtractor.Align(image, rectangle, angle);
            return _faceEmbedder.Forward(aligned);
        }

        return new float[512];
    }

    /// <summary>
    /// Method that initializes capturing video from the camera.
    /// </summary>
    /// <returns>Returns true if the camera was opened successfully, false otherwise.</returns>
    public bool CameraInitialize()
    {
        capture = new VideoCapture(0);

        return capture.IsOpened;
    }


    /// <summary>
    /// A method that detects a face in a camera image using a neural network.
    /// </summary>
    /// <returns>Returns a Mat containing the image of the face, or null if no face was found.</returns>
    /// <exception cref="Exception">Throws an exception if the camera image is empty.</exception>
    public Bitmap FaceDetect()
    {
        using var frame = new Mat();

        capture.Read(frame);

        if (frame.IsEmpty)
            throw new Exception("Received image is empty");

        return frame.ToImage<Gray, byte>().ToBitmap();
    }

    /// <summary>
    /// A method that recognizes a face using the FisherFaceRecognizer algorithm.
    /// </summary>
    /// <returns>Returns true if the face is recognized, false otherwise.</returns>
    /// <exception cref="Exception">Throws an exception if the recognition model is not trained.</exception>
    public bool Recognize()
    {
        using var face = FaceDetect();
        using var bitmap = new Bitmap(face);
        var embedding = GetEmbedding(bitmap);
        var proto = embeddings.FromSimilarity(embedding);
        var label = proto.Item1;
        var similarity = proto.Item2;

        if (!String.IsNullOrEmpty(label) && similarity > 0.6f)
            return true;

        return false;
    }

    /// <summary>
    /// A method that registers a face and saves its image in the TrainedFaces folder.
    /// </summary>
    /// <param name="name">Name of the person.</param>
    /// <param name="count">Number of faces to register.</param>
    public void RegisterFace(string name, int count = 1)
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

            FaceDetect().Save(filename);
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
            string[] regFaces = Directory.GetFiles(DbFolderName, "*.jpg", SearchOption.AllDirectories);

            if (regFaces.Length != 0)
            {
                foreach (var regFace in regFaces)
                {
                    using var bitmap = new Bitmap(regFace);
                    var embedding = GetEmbedding(bitmap);
                    var name = Path.GetFileNameWithoutExtension(regFace);
                    embeddings.Add(embedding, name);
                }
            }
        }
    }
}