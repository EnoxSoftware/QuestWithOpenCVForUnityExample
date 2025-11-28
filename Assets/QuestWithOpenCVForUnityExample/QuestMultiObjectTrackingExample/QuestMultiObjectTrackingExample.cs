#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using OpenCVForUnity.UnityIntegration.MOT;
using OpenCVForUnity.UnityIntegration.MOT.ByteTrack;
using OpenCVForUnity.UnityIntegration.Worker.DataStruct;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace QuestWithOpenCVForUnityExample
{
    /// <summary>
    /// Quest Multi Object Tracking Example
    /// An example of using OpenCV dnn module with YOLOX Object Detection on MetaQuest.
    /// Referring to :
    /// https://github.com/opencv/opencv_zoo/tree/master/models/object_detection_yolox
    /// https://github.com/Megvii-BaseDetection/YOLOX
    /// https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
    ///
    /// [Tested Models]
    /// yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
    /// yolox_tiny.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
    /// yolox_s.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx
    /// </summary>
    [RequireComponent(typeof(QuestPassthrough2MatHelper))]
    public class QuestMultiObjectTrackingExample : MonoBehaviour
    {
        // Public Fields
        [Header("UI")]
        public Toggle UseAsyncInferenceToggle;
        public bool UseAsyncInference = false;
        public Toggle ShowPassthroughImageToggle;
        public bool ShowPassthroughImage = false;
        public Toggle ShowObjectDetectorResultToggle;
        public bool ShowObjectDetectorResult;
        public Toggle EnableByteTrackToggle;
        public bool EnableByteTrack;

        [Header("Model Settings")]
        [Tooltip("Path to a binary file of model contains trained weights.")]
        public string Model = "OpenCVForUnityExamples/dnn/yolox_tiny.onnx";

        [Tooltip("Optional path to a text file with names of classes to label detected objects.")]
        public string Classes = "OpenCVForUnityExamples/dnn/coco.names";

        [Tooltip("Confidence threshold.")]
        public float ConfThreshold = 0.25f;

        [Tooltip("Non-maximum suppression threshold.")]
        public float NmsThreshold = 0.45f;

        [Tooltip("Maximum detections per image.")]
        public int TopK = 300;

        [Tooltip("Preprocess input image by resizing to a specific width.")]
        public int InpWidth = 416;

        [Tooltip("Preprocess input image by resizing to a specific height.")]
        public int InpHeight = 416;

        [Space(10)]

        [HeaderAttribute("Debug")]

        public Text RenderFPS;
        public Text VideoFPS;
        public Text TrackFPS;
        public Text DebugStr;

        // Private Fields
        private YOLOXObjectDetector _objectDetector;
        private BYTETracker _byteTracker;
        private BYTETrackInfoVisualizer _byteTrackInfoVisualizer;
        private string _classesFilepath;
        private string _modelFilepath;

        private Texture2D _texture;
        private Renderer _quadRenderer;
        private QuestPassthrough2MatHelper _webCamTextureToMatHelper;
        private Mat _bgrMat;

        private CancellationTokenSource _cts = new CancellationTokenSource();

        private Mat _bgrMatForAsync;
        private Mat _latestDetectedObjects;
        private Task _inferenceTask;
        private readonly Queue<Action> _mainThreadQueue = new();
        private readonly object _queueLock = new();

        // Unity Lifecycle Methods
        private async void Start()
        {
            _webCamTextureToMatHelper = gameObject.GetComponent<QuestPassthrough2MatHelper>();
            _webCamTextureToMatHelper.OutputColorFormat = Source2MatHelperColorFormat.RGBA;

            // Update GUI state
            UseAsyncInferenceToggle.isOn = UseAsyncInference;
            ShowPassthroughImageToggle.isOn = ShowPassthroughImage;
            ShowObjectDetectorResultToggle.isOn = ShowObjectDetectorResult;
            EnableByteTrackToggle.isOn = EnableByteTrack;

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            DebugUtils.AddDebugStr("Preparing file access...");

            if (!string.IsNullOrEmpty(Classes))
            {
                _classesFilepath = await OpenCVEnv.GetFilePathTaskAsync(Classes, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_classesFilepath)) Debug.Log("The file:" + Classes + " did not exist.");
            }
            if (!string.IsNullOrEmpty(Model))
            {
                _modelFilepath = await OpenCVEnv.GetFilePathTaskAsync(Model, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_modelFilepath)) Debug.Log("The file:" + Model + " did not exist.");
            }

            DebugUtils.ClearDebugStr();

            Run();
        }

        private void Run()
        {
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            OpenCVDebug.SetDebugMode(true);

            if (string.IsNullOrEmpty(_modelFilepath))
            {
                Debug.LogError("model: " + Model + " or " + "classes: " + Classes + " is not loaded. Please use [Tools] > [OpenCV for Unity] > [Setup Tools] > [Example Assets Downloader]to download the asset files required for this example scene, and then move them to the \"Assets/StreamingAssets\" folder.");
            }
            else
            {
                _objectDetector = new YOLOXObjectDetector(_modelFilepath, _classesFilepath, new Size(InpWidth, InpHeight), ConfThreshold, NmsThreshold, TopK);
                _byteTrackInfoVisualizer = new BYTETrackInfoVisualizer(_classesFilepath);
            }

            _webCamTextureToMatHelper.Initialize();
        }

        // Public Methods
        /// <summary>
        /// Raises the source to mat helper initialized event.
        /// </summary>
        public void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat rgbaMat = _webCamTextureToMatHelper.GetMat();

            _texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);
            _texture.wrapMode = TextureWrapMode.Clamp;
            _quadRenderer = gameObject.GetComponent<Renderer>() as Renderer;
            _quadRenderer.sharedMaterial.SetTexture("_MainTex", _texture);

            //Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            DebugUtils.AddDebugStr(_webCamTextureToMatHelper.OutputColorFormat.ToString() + " " + _webCamTextureToMatHelper.GetWidth() + " x " + _webCamTextureToMatHelper.GetHeight() + " : " + _webCamTextureToMatHelper.GetFPS());

            Matrix4x4 projectionMatrix;
#if UNITY_ANDROID && !DISABLE_QUESTPASSTHROUGH_API
            projectionMatrix = _webCamTextureToMatHelper.GetProjectionMatrix();
            _quadRenderer.sharedMaterial.SetMatrix("_CameraProjectionMatrix", projectionMatrix);
#else
            projectionMatrix = Matrix4x4.identity;
            projectionMatrix.m00 = 1.35818000f;
            projectionMatrix.m01 = 0.00000000f;
            projectionMatrix.m02 = 0.00082388f;
            projectionMatrix.m03 = 0.00000000f;
            projectionMatrix.m10 = 0.00000000f;
            projectionMatrix.m11 = 1.81090700f;
            projectionMatrix.m12 = 0.00302397f;
            projectionMatrix.m13 = 0.00000000f;
            projectionMatrix.m20 = 0.00000000f;
            projectionMatrix.m21 = 0.00000000f;
            projectionMatrix.m22 = -1.00020000f;
            projectionMatrix.m23 = -0.02000200f;
            projectionMatrix.m30 = 0.00000000f;
            projectionMatrix.m31 = 0.00000000f;
            projectionMatrix.m32 = -1.00000000f;
            projectionMatrix.m33 = 0.00000000f;
            _quadRenderer.sharedMaterial.SetMatrix("_CameraProjectionMatrix", projectionMatrix);
#endif

            float halfOfVerticalFov = Mathf.Atan(1.0f / projectionMatrix.m11);
            float aspectRatio = (1.0f / Mathf.Tan(halfOfVerticalFov)) / projectionMatrix.m00;
            Debug.Log("halfOfVerticalFov " + halfOfVerticalFov);
            Debug.Log("aspectRatio " + aspectRatio);

            _bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
            _bgrMatForAsync = new Mat();

            int fps = 24;// The Meta Quest passthrough camera returns 0 to webCamTexture.requestedFPS, so specify it directly.
            _byteTracker = new BYTETracker(fps, 30, mot20: false);
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            if (_inferenceTask != null && !_inferenceTask.IsCompleted) _inferenceTask.Wait(500);

            _bgrMat?.Dispose(); _bgrMat = null;

            _bgrMatForAsync?.Dispose(); _bgrMatForAsync = null;
            _latestDetectedObjects?.Dispose(); _latestDetectedObjects = null;

            _byteTracker?.Dispose(); _byteTracker = null;

            if (_texture != null) Texture2D.Destroy(_texture); _texture = null;

            if (DebugStr != null) DebugStr.text = string.Empty;
            DebugUtils.ClearDebugStr();
        }

        /// <summary>
        /// Raises the source to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        /// <param name="message">Message.</param>
        public void OnSourceToMatHelperErrorOccurred(Source2MatHelperErrorCode errorCode, string message)
        {
            Debug.Log("OnSourceToMatHelperErrorOccurred " + errorCode + ":" + message);
        }

        private void Update()
        {
            ProcessMainThreadQueue();

            if (_webCamTextureToMatHelper.IsPlaying() && _webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                DebugUtils.VideoTick();

                Mat rgbaMat = _webCamTextureToMatHelper.GetMat();

                if (_objectDetector == null)
                {
                    Imgproc.putText(rgbaMat, "model file is not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                    Imgproc.putText(rgbaMat, "Please read console message.", new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
                }
                else
                {
                    Imgproc.cvtColor(rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);

                    if (UseAsyncInference)
                    {
                        // asynchronous execution

                        if (_inferenceTask == null || _inferenceTask.IsCompleted)
                        {
                            _bgrMat.copyTo(_bgrMatForAsync); // for asynchronous execution, deep copy
                            _inferenceTask = Task.Run(async () =>
                            {
                                try
                                {
                                    // Object detector inference
                                    var newObjects = await _objectDetector.DetectAsync(_bgrMatForAsync);
                                    RunOnMainThread(() =>
                                        {
                                            _latestDetectedObjects?.Dispose();
                                            _latestDetectedObjects = newObjects;
                                        });
                                }
                                catch (OperationCanceledException ex)
                                {
                                    Debug.Log($"Inference canceled: {ex}");
                                }
                                catch (Exception ex)
                                {
                                    Debug.LogError($"Inference error: {ex}");
                                }
                            });
                        }

                        if (ShowPassthroughImage)
                        {
                            Imgproc.cvtColor(_bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);
                        }
                        else
                        {
                            // Fill with transparent color
                            rgbaMat.setTo(new Scalar(0, 0, 0, 0));
                        }

                        if (_latestDetectedObjects != null)
                        {
                            if (ShowObjectDetectorResult)
                                _objectDetector.Visualize(rgbaMat, _latestDetectedObjects, false, true);

                            if (EnableByteTrack)
                            {
                                BBox[] inputs = ConvertToBBoxes(_latestDetectedObjects);
                                _byteTracker.Update(inputs);
                            }

                            _latestDetectedObjects.Dispose();
                            _latestDetectedObjects = null;
                        }

                        if (EnableByteTrack)
                        {
                            // Visualize tracking results (tracks are maintained even when no detection is available)
                            BYTETrackInfo[] outputs = _byteTracker.GetActiveTrackInfos();
                            _byteTrackInfoVisualizer.Visualize(rgbaMat, outputs, false, true);
                        }
                    }
                    else
                    {
                        // synchronous execution

                        // Object detector inference
                        using (Mat objects = _objectDetector.Detect(_bgrMat))
                        {
                            if (ShowPassthroughImage)
                            {
                                Imgproc.cvtColor(_bgrMat, rgbaMat, Imgproc.COLOR_BGR2RGBA);
                            }
                            else
                            {
                                // Fill with transparent color
                                rgbaMat.setTo(new Scalar(0, 0, 0, 0));
                            }

                            if (ShowObjectDetectorResult)
                                _objectDetector.Visualize(rgbaMat, objects, false, true);

                            if (EnableByteTrack)
                            {
                                BBox[] inputs = ConvertToBBoxes(objects);
                                _byteTracker.Update(inputs);
                                BYTETrackInfo[] outputs = _byteTracker.GetActiveTrackInfos();
                                _byteTrackInfoVisualizer.Visualize(rgbaMat, outputs, false, true);
                            }
                        }
                    }
                }

                DebugUtils.TrackTick();

                OpenCVMatUtils.MatToTexture2D(rgbaMat, _texture);
            }

            if (_webCamTextureToMatHelper.IsPlaying())
            {
                Matrix4x4 cameraToWorldMatrix = _webCamTextureToMatHelper.GetCameraToWorldMatrix();
                Matrix4x4 worldToCameraMatrix = cameraToWorldMatrix.inverse;

                _quadRenderer.sharedMaterial.SetMatrix("_WorldToCameraMatrix", worldToCameraMatrix);

                // Position the canvas object slightly in front
                // of the real world web camera.
                Vector3 position = cameraToWorldMatrix.GetColumn(3) - cameraToWorldMatrix.GetColumn(2) * 2.2f;

                // Rotate the canvas object so that it faces the user.
                Quaternion rotation = Quaternion.LookRotation(-cameraToWorldMatrix.GetColumn(2), cameraToWorldMatrix.GetColumn(1));

                gameObject.transform.position = position;
                gameObject.transform.rotation = rotation;
            }
        }

        private void LateUpdate()
        {
            DebugUtils.RenderTick();
            float renderDeltaTime = DebugUtils.GetRenderDeltaTime();
            float videoDeltaTime = DebugUtils.GetVideoDeltaTime();
            float trackDeltaTime = DebugUtils.GetTrackDeltaTime();

            if (RenderFPS != null)
            {
                RenderFPS.text = string.Format("Render: {0:0.0} ms ({1:0.} fps)", renderDeltaTime, 1000.0f / renderDeltaTime);
            }
            if (VideoFPS != null)
            {
                VideoFPS.text = string.Format("Video: {0:0.0} ms ({1:0.} fps)", videoDeltaTime, 1000.0f / videoDeltaTime);
            }
            if (TrackFPS != null)
            {
                TrackFPS.text = string.Format("Track:   {0:0.0} ms ({1:0.} fps)", trackDeltaTime, 1000.0f / trackDeltaTime);
            }
            if (DebugStr != null)
            {
                if (DebugUtils.GetDebugStrLength() > 0)
                {
                    if (DebugStr.preferredHeight >= DebugStr.rectTransform.rect.height)
                        DebugStr.text = string.Empty;

                    DebugStr.text += DebugUtils.GetDebugStr();
                    DebugUtils.ClearDebugStr();
                }
            }
        }

        private void OnDestroy()
        {
            _webCamTextureToMatHelper?.Dispose();

            _objectDetector?.Dispose();
            _byteTracker?.Dispose();
            _byteTrackInfoVisualizer?.Dispose();

            OpenCVDebug.SetDebugMode(false);

            _cts?.Dispose();
        }

        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("QuestWithOpenCVForUnityExample");
        }

        /// <summary>
        /// Raises the play button click event.
        /// </summary>
        public void OnPlayButtonClick()
        {
            _webCamTextureToMatHelper.Play();
        }

        /// <summary>
        /// Raises the pause button click event.
        /// </summary>
        public void OnPauseButtonClick()
        {
            _webCamTextureToMatHelper.Pause();
        }

        /// <summary>
        /// Raises the stop button click event.
        /// </summary>
        public void OnStopButtonClick()
        {
            _webCamTextureToMatHelper.Stop();
        }

        /// <summary>
        /// Raises the change camera button click event.
        /// </summary>
        public void OnChangeCameraButtonClick()
        {
            _webCamTextureToMatHelper.RequestedIsFrontFacing = !_webCamTextureToMatHelper.RequestedIsFrontFacing;
        }

        /// <summary>
        /// Raises the use async inference toggle value changed event.
        /// </summary>
        public void OnUseAsyncInferenceToggleValueChanged()
        {
            if (UseAsyncInferenceToggle.isOn != UseAsyncInference)
            {
                // Wait for inference to complete before changing the toggle
                if (_inferenceTask != null && !_inferenceTask.IsCompleted) _inferenceTask.Wait(500);

                UseAsyncInference = UseAsyncInferenceToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the show passthrough image toggle value changed event.
        /// </summary>
        public void OnShowPassthroughImageToggleValueChanged()
        {
            if (ShowPassthroughImageToggle.isOn != ShowPassthroughImage)
            {
                ShowPassthroughImage = ShowPassthroughImageToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the show object detector result toggle value changed event.
        /// </summary>
        public void OnShowObjectDetectorResultToggleValueChanged()
        {
            if (ShowObjectDetectorResultToggle.isOn != ShowObjectDetectorResult)
            {
                ShowObjectDetectorResult = ShowObjectDetectorResultToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the enable byte track toggle value changed event.
        /// </summary>
        public void OnEnableByteTrackToggleValueChanged()
        {
            if (EnableByteTrackToggle.isOn != EnableByteTrack)
            {
                EnableByteTrack = EnableByteTrackToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the reset trackers button click event.
        /// </summary>
        public void OnResetTrackersButtonClick()
        {
            ResetTrackers();
        }

        // Private Methods
        private void ResetTrackers()
        {
            _byteTracker?.Reset();
        }

        private BBox[] ConvertToBBoxes(Mat result)
        {
            if (result.empty() || result.cols() < 6)
                return new BBox[0];

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<ObjectDetectionData> data = _objectDetector.ToStructuredDataAsSpan(result);
#else
            ObjectDetectionData[] data = _objectDetector.ToStructuredData(result);
#endif

            BBox[] inputs = new BBox[data.Length];
            for (int i = 0; i < data.Length; ++i)
            {
                ref readonly var d = ref data[i];
                inputs[i] = new BBox(d);
            }

            return inputs;
        }

        private void RunOnMainThread(Action action)
        {
            if (action == null) return;

            lock (_queueLock)
            {
                _mainThreadQueue.Enqueue(action);
            }
        }

        private void ProcessMainThreadQueue()
        {
            while (true)
            {
                Action action = null;
                lock (_queueLock)
                {
                    if (_mainThreadQueue.Count == 0)
                        break;

                    action = _mainThreadQueue.Dequeue();
                }

                try { action?.Invoke(); }
                catch (Exception ex) { Debug.LogException(ex); }
            }
        }
    }
}

#endif
