using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceAuth;

public class FaceAuthProvider
{
    #region Vaiable
    private int[] FaceLabels;
    private float FaceDistance = -1;
    private int FaceThreshold = 3500;
    private const string dbFolderName = "TrainedFaces";
    private FaceRecognizer recognizer = new FisherFaceRecognizer(0, 3500);
    FaceRecognizer.PredictionResult result = new();
    private List<Image<Gray, byte>> imgList = new();
    private bool _isTrained;

    #endregion

    public bool Recognize(Mat face, int FaceThresh = -1)
    {
        if (_isTrained)
        {
            result = recognizer.Predict(face.ToImage<Gray, byte>());

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
                    using VectorOfInt vectorOfInt = new VectorOfInt(FaceLabels = Enumerable.Range(0, files.Length).ToArray());
                    vectorOfMat.Push(imgList.ToArray());

                    try
                    {
                        recognizer.Train(vectorOfMat, vectorOfInt);
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