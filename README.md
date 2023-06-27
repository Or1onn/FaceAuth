# FaceAuth

FaceAuth is a library that provides **the easiest** way to use face authentication in your .NET project in just 4 lines of code.

---

## Installation

To install, enter the command in the CLI

```bash 
dotnet add package FaceAuth
```
---

## Using

```csharp
using FaceAuth;

var auth = new FaceAuthProvider();

// Check and Initialize out camera
if (auth.CameraInitialize())
{
    // Initialize FaceRecognizer model
    auth.LoadFaceRecognizer();
    
    //Register
    auth.RegisterIFace("The name of the person you are registering", 10);

    // Recognize face
    auth.Recognize();
}
else
{
    Console.WriteLine("Your Camera Not Found!");
}
```

---

## Advices

* For a big job, you need from **50-200** images of one person in **different head positions.**
* You can put auth.Recognize() in a loop of 10 iterations and get at least 1 true for successful authentication
```csharp
    // Recognize face
    for (int i = 0; i < 10; i++)
    {
        if (auth.Recognize())
        {
            Console.WriteLine("Successful authentication");
            return;
        }
    }

    Console.WriteLine("Failed authentication");
```
* It is desirable to do Recognize() not in an infinite loop, for example, while observing the efficient authentication algorithm.
* It is recommended to make an anti-spam system to protect against hacking.
```csharp
bool TryRecognize(FaceAuthProvider _auth, ref int _recognizeCount)
{
    _recognizeCount++;
    return _auth.Recognize();
}

int recognizeCount = 0;

for (int i = 0; i < 10; i++)
{
    if (recognizeCount <= 5)
    {
        TryRecognize(auth, ref recognizeCount);
    }
    else
    {
        Console.WriteLine("You cant recognize more");
        return;
    }
}
```