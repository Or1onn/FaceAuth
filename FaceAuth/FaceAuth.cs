using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceAuth;

public class FaceAuth
{
    #region Vaiable
    
    private float FaceDistance = -1;
    private  int FaceThreshold = 3000;
    private string dbFolderName = "face_db";
    private FaceRecognizer recognizer = new FisherFaceRecognizer(0, 3500); //4000
    private List<Image<Gray, byte>> imgList = new();
    private List<int> imgIds = new(){};
    
    #endregion

    public void LoadFaceRecognizer()
    {
        if (!Directory.Exists(dbFolderName))
        {
            Directory.CreateDirectory(dbFolderName);
            Directory.CreateDirectory(AppDomain.CurrentDomain.BaseDirectory + "\\" + dbFolderName);
        }
        else
        {
            string[] files = Directory.GetFiles(dbFolderName, "*.jpg", SearchOption.AllDirectories);

            Parallel.ForEach(files, file => { imgList.Add(new Image<Gray, byte>(file)); });

            using VectorOfMat vectorOfMat = new VectorOfMat();
            using VectorOfInt vectorOfInt = new VectorOfInt(imgIds.ToArray());
            vectorOfMat.Push(imgList.ToArray());

            recognizer.Train(vectorOfMat, vectorOfInt);
        }
    }
}