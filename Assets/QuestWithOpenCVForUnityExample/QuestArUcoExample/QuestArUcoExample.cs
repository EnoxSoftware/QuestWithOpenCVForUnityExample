using System;
using System.Collections.Generic;
using System.Threading;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.AR;
using OpenCVForUnity.UnityIntegration.Helper.Optimization;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace QuestWithOpenCVForUnityExample
{
    /// <summary>
    /// Quest ArUco Example
    /// An example of marker based AR using OpenCVForUnity on MetaQuest.
    /// Referring to https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp.
    /// </summary>
    [RequireComponent(typeof(QuestPassthrough2MatHelper), typeof(ImageOptimizationHelper))]
    public class QuestArUcoExample : MonoBehaviour
    {
        // Enums
        /// <summary>
        /// Marker type enum
        /// </summary>
        public enum MarkerType
        {
            CanonicalMarker,
            GridBoard,
            ChArUcoBoard,
            ChArUcoDiamondMarker
        }

        /// <summary>
        /// ArUco dictionary enum
        /// </summary>
        public enum ArUcoDictionary
        {
            DICT_4X4_50 = Objdetect.DICT_4X4_50,
            DICT_4X4_100 = Objdetect.DICT_4X4_100,
            DICT_4X4_250 = Objdetect.DICT_4X4_250,
            DICT_4X4_1000 = Objdetect.DICT_4X4_1000,
            DICT_5X5_50 = Objdetect.DICT_5X5_50,
            DICT_5X5_100 = Objdetect.DICT_5X5_100,
            DICT_5X5_250 = Objdetect.DICT_5X5_250,
            DICT_5X5_1000 = Objdetect.DICT_5X5_1000,
            DICT_6X6_50 = Objdetect.DICT_6X6_50,
            DICT_6X6_100 = Objdetect.DICT_6X6_100,
            DICT_6X6_250 = Objdetect.DICT_6X6_250,
            DICT_6X6_1000 = Objdetect.DICT_6X6_1000,
            DICT_7X7_50 = Objdetect.DICT_7X7_50,
            DICT_7X7_100 = Objdetect.DICT_7X7_100,
            DICT_7X7_250 = Objdetect.DICT_7X7_250,
            DICT_7X7_1000 = Objdetect.DICT_7X7_1000,
            DICT_ARUCO_ORIGINAL = Objdetect.DICT_ARUCO_ORIGINAL,
        }

        // Public Fields
        [HeaderAttribute("Preview")]
        public GameObject PreviewQuad;
        public Toggle DisplayCameraPreviewToggle;
        public bool DisplayCameraPreview;

        [HeaderAttribute("Detection")]
        public bool EnableDetection = true;
        public Toggle EnableDownScaleToggle;
        public bool EnableDownScale;

        [HeaderAttribute("AR")]
        public bool ApplyEstimationPose = true;
        public Dropdown DictionaryIdDropdown;
        public ArUcoDictionary DictionaryId = ArUcoDictionary.DICT_6X6_250;
        public Toggle EnableLowPassFilterToggle;
        public bool EnableLowPassFilter = false;
        public Toggle EnableSmoothingFilterToggle;
        public bool EnableSmoothingFilter = false;
        public Toggle EnableSOLVEPNP_ITERATIVEToggle;
        public bool EnableSOLVEPNP_ITERATIVE = false;

        [HeaderAttribute("Debug")]
        public Text RenderFPS;
        public Text RideoFPS;
        public Text TrackFPS;
        public Text DebugStr;

        [Space(10)]

        [Tooltip("The length of the markers' side. Normally, unit is meters.")]
        public float MarkerLength = 0.188f;
        public ARHelper ArHelper;
        public GameObject ArCubePrefab;

        // Private Fields
        private MarkerType _selectedMarkerType = MarkerType.CanonicalMarker;
        private Texture2D _texture;
        private QuestPassthrough2MatHelper _webCamTextureToMatHelper;
        private ImageOptimizationHelper _imageOptimizationHelper;
        private Mat _downScaleMat;
        private float _downscaleRatio;
        private Mat _rgbMatForPreview;
        private Mat _camMatrix;
        private MatOfDouble _distCoeffs;

        private Mat _downScaleMatForWorker; // Thread-safe copy for worker thread
        private Mat _undistortedRgbMatForWorker; // Thread-safe undistorted image for worker thread

        // Thread-safe copies of camera parameters for worker thread (read-only, can be shared)
        private Mat _camMatrixForWorker;
        private MatOfDouble _distCoeffsForWorker;

        // for CanonicalMarker.
        private Dictionary _dictionary;
        private ArucoDetector _arucoDetector;

        private Dictionary<ArUcoIdentifier, ARGameObject> _arGameObjectCache = new Dictionary<ArUcoIdentifier, ARGameObject>();

        // Detection results for thread-safe transfer to main thread
        private struct DetectionResult
        {
            public int MarkerId;
            public Vector2[] ImagePoints;
            public Vector3[] ObjectPoints;
        }
        private List<DetectionResult> _detectionResults = new List<DetectionResult>();

        private readonly static Queue<Action> _executeOnMainThread = new Queue<Action>();
        private readonly System.Object _sync = new System.Object();

        private bool _isThreadRunning = false;
        private bool IsThreadRunning
        {
            get
            {
                lock (_sync)
                    return _isThreadRunning;
            }
            set
            {
                lock (_sync)
                    _isThreadRunning = value;
            }
        }

        private bool _isDetecting = false;
        private bool IsDetecting
        {
            get
            {
                lock (_sync)
                    return _isDetecting;
            }
            set
            {
                lock (_sync)
                    _isDetecting = value;
            }
        }

        // Use this for initialization
        private void Start()
        {
            _imageOptimizationHelper = gameObject.GetComponent<ImageOptimizationHelper>();
            _webCamTextureToMatHelper = gameObject.GetComponent<QuestPassthrough2MatHelper>();
            _webCamTextureToMatHelper.OutputColorFormat = Source2MatHelperColorFormat.GRAY;
            _webCamTextureToMatHelper.Initialize();

            // Update GUI state
            DictionaryIdDropdown.value = (int)DictionaryId;
            DisplayCameraPreviewToggle.isOn = DisplayCameraPreview;
            EnableDownScaleToggle.isOn = EnableDownScale;
            EnableLowPassFilterToggle.isOn = EnableLowPassFilter;
            EnableSmoothingFilterToggle.isOn = EnableSmoothingFilter;
            EnableSOLVEPNP_ITERATIVEToggle.isOn = EnableSOLVEPNP_ITERATIVE;
        }

        // Update is called once per frame
        private void Update()
        {
            // Get the latest pose matrix of the MetaQuest Passthrough camera and apply it to the dummy AR camera's Transform.
            Matrix4x4 cameraLocalToWorldMatrix = _webCamTextureToMatHelper.GetCameraToWorldMatrix() * Matrix4x4.Scale(new Vector3(1, 1, -1));
            OpenCVARUtils.SetTransformFromMatrix(ArHelper.ARCamera.transform, ref cameraLocalToWorldMatrix);

            lock (_executeOnMainThread)
            {
                while (_executeOnMainThread.Count > 0)
                {
                    _executeOnMainThread.Dequeue().Invoke();
                }
            }

            if (_webCamTextureToMatHelper.IsPlaying() && _webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                DebugUtils.VideoTick();

                if (EnableDetection && !IsDetecting)
                {
                    IsDetecting = true;

                    Mat grayMat = _webCamTextureToMatHelper.GetMat();

                    if (EnableDownScale)
                    {
                        _downScaleMat = _imageOptimizationHelper.GetDownScaleMat(grayMat);
                        _downscaleRatio = _imageOptimizationHelper.DownscaleRatio;
                    }
                    else
                    {
                        _downScaleMat = grayMat;
                        _downscaleRatio = 1.0f;
                    }

                    // Create or update thread-safe copy for worker thread
                    lock (_sync)
                    {
                        if (_downScaleMat != null && !_downScaleMat.empty())
                        {
                            // Check if _downScaleMatForWorker needs to be recreated (size changed)
                            if (_downScaleMatForWorker == null || _downScaleMatForWorker.empty() ||
                                _downScaleMatForWorker.width() != _downScaleMat.width() ||
                                _downScaleMatForWorker.height() != _downScaleMat.height() ||
                                _downScaleMatForWorker.type() != _downScaleMat.type())
                            {
                                _downScaleMatForWorker?.Dispose();
                                _downScaleMatForWorker = new Mat(_downScaleMat.rows(), _downScaleMat.cols(), _downScaleMat.type());
                            }
                            // Copy data using copyTo (more efficient than clone when Mat already exists)
                            _downScaleMat.copyTo(_downScaleMatForWorker);
                        }
                    }

                    StartThread(ThreadWorker);
                }
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
            if (RideoFPS != null)
            {
                RideoFPS.text = string.Format("Video: {0:0.0} ms ({1:0.} fps)", videoDeltaTime, 1000.0f / videoDeltaTime);
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
            _imageOptimizationHelper?.Dispose();
        }

        /// <summary>
        /// Raises the source to mat helper initialized event.
        /// </summary>
        public void OnSourceToMatHelperInitialized()
        {
            Debug.Log("OnSourceToMatHelperInitialized");

            Mat grayMat = _webCamTextureToMatHelper.GetMat();

            float rawFrameWidth = grayMat.width();
            float rawFrameHeight = grayMat.height();

            if (EnableDownScale)
            {
                _downScaleMat = _imageOptimizationHelper.GetDownScaleMat(grayMat);
                _downscaleRatio = _imageOptimizationHelper.DownscaleRatio;
            }
            else
            {
                _downScaleMat = grayMat;
                _downscaleRatio = 1.0f;
            }

            float width = _downScaleMat.width();
            float height = _downScaleMat.height();

            _texture = new Texture2D((int)width, (int)height, TextureFormat.RGB24, false);
            PreviewQuad.GetComponent<MeshRenderer>().material.mainTexture = _texture;
            PreviewQuad.SetActive(DisplayCameraPreview);


            //Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);


            DebugUtils.AddDebugStr(_webCamTextureToMatHelper.OutputColorFormat.ToString() + " " + _webCamTextureToMatHelper.GetWidth() + " x " + _webCamTextureToMatHelper.GetHeight() + " : " + _webCamTextureToMatHelper.GetFPS());
            if (EnableDownScale)
                DebugUtils.AddDebugStr("enableDownScale = true: " + _downscaleRatio + " / " + width + " x " + height);


            // create camera matrix and dist coeffs.
#if UNITY_ANDROID && !DISABLE_QUESTPASSTHROUGH_API

            Matrix4x4 projectionMatrix = _webCamTextureToMatHelper.GetProjectionMatrix();
            Matrix4x4 cameraMatrix = OpenCVARUtils.CalculateCameraMatrixValuesFromProjectionMatrix(projectionMatrix, width, height);
            _camMatrix = CreateCameraMatrix(cameraMatrix.m00, cameraMatrix.m11, cameraMatrix.m02, cameraMatrix.m12);
            _distCoeffs = new MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0);

            Debug.Log("Created CameraParameters from ProjectionMatrix on device.");

            DebugUtils.AddDebugStr("Created CameraParameters from ProjectionMatrix on device.");

#else

            Matrix4x4 projectionMatrix;
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

            Matrix4x4 cameraMatrix = OpenCVARUtils.CalculateCameraMatrixValuesFromProjectionMatrix(projectionMatrix, 1280 / _downscaleRatio, 960 / _downscaleRatio);
            _camMatrix = CreateCameraMatrix(cameraMatrix.m00, cameraMatrix.m11, cameraMatrix.m02, cameraMatrix.m12);
            _distCoeffs = new MatOfDouble(0.0, 0.0, 0.0, 0.0, 0.0);

            Debug.Log("Created a dummy CameraParameters (1280x960).");

            DebugUtils.AddDebugStr("Created a dummy CameraParameters (1280x960).");

#endif

            Debug.Log("camMatrix " + _camMatrix.dump());
            Debug.Log("distCoeffs " + _distCoeffs.dump());

            DebugUtils.AddDebugStr("camMatrix " + _camMatrix.dump());
            DebugUtils.AddDebugStr("distCoeffs " + _distCoeffs.dump());


            //Calibration camera
            Size imageSize = new Size(width, height);
            double apertureWidth = 0;
            double apertureHeight = 0;
            double[] fovx = new double[1];
            double[] fovy = new double[1];
            double[] focalLength = new double[1];
            Point principalPoint = new Point(0, 0);
            double[] aspectratio = new double[1];

            Calib3d.calibrationMatrixValues(_camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

            Debug.Log("imageSize " + imageSize.ToString());
            Debug.Log("apertureWidth " + apertureWidth);
            Debug.Log("apertureHeight " + apertureHeight);
            Debug.Log("fovx " + fovx[0]);
            Debug.Log("fovy " + fovy[0]);
            Debug.Log("focalLength " + focalLength[0]);
            Debug.Log("principalPoint " + principalPoint.ToString());
            Debug.Log("aspectratio " + aspectratio[0]);

            _dictionary = Objdetect.getPredefinedDictionary((int)DictionaryId);

            // Create thread-safe copies of camera parameters for worker thread
            _camMatrixForWorker = _camMatrix.clone();
            _distCoeffsForWorker = new MatOfDouble(_distCoeffs);

            // Create thread-safe undistorted image Mat for worker thread
            _undistortedRgbMatForWorker = new Mat();

            DetectorParameters detectorParams = new DetectorParameters();
            detectorParams.set_minDistanceToBorder(3);
            detectorParams.set_useAruco3Detection(true);
            detectorParams.set_cornerRefinementMethod(Objdetect.CORNER_REFINE_SUBPIX);
            detectorParams.set_minSideLengthCanonicalImg(16);
            detectorParams.set_errorCorrectionRate(0.8);
            RefineParameters refineParameters = new RefineParameters(10f, 3f, true);
            _arucoDetector = new ArucoDetector(_dictionary, detectorParams, refineParameters);


            // If the WebCam is front facing, flip the Mat horizontally. Required for successful detection of AR markers.
            _webCamTextureToMatHelper.FlipHorizontal = _webCamTextureToMatHelper.IsFrontFacing();

            _rgbMatForPreview = new Mat();

            // Initialize ARHelper.
            if (ArHelper != null)
            {
                ArHelper.Initialize();
                // Set ARCamera parameters.
                ArHelper.ARCamera.SetCamMatrix(_camMatrix);
                ArHelper.ARCamera.SetDistCoeffs(_distCoeffs);
                ArHelper.ARCamera.SetARCameraParameters(Screen.width, Screen.height, (int)width, (int)height, Vector2.zero, new Vector2(1.0f, 1.0f));
            }
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            StopThread();
            lock (_executeOnMainThread)
            {
                _executeOnMainThread.Clear();
            }
            IsDetecting = false;

            _arucoDetector?.Dispose(); _arucoDetector = null;

            _dictionary?.Dispose(); _dictionary = null;

            _camMatrixForWorker?.Dispose(); _camMatrixForWorker = null;
            _distCoeffsForWorker?.Dispose(); _distCoeffsForWorker = null;

            _downScaleMatForWorker?.Dispose(); _downScaleMatForWorker = null;
            _undistortedRgbMatForWorker?.Dispose(); _undistortedRgbMatForWorker = null;

            if (ArHelper != null)
            {
                RemoveAllARGameObject(ArHelper.ARGameObjects);
                ArHelper.Dispose();
            }

            _rgbMatForPreview?.Dispose();

            if (DebugStr != null)
            {
                DebugStr.text = string.Empty;
            }
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
        /// Raises the display camera preview toggle value changed event.
        /// </summary>
        public void OnDisplayCamreaPreviewToggleValueChanged()
        {
            DisplayCameraPreview = DisplayCameraPreviewToggle.isOn;

            PreviewQuad.SetActive(DisplayCameraPreview);
        }

        /// <summary>
        /// Raises the enable downscale toggle value changed event.
        /// </summary>
        public void OnEnableDownScaleToggleValueChanged()
        {
            EnableDownScale = EnableDownScaleToggle.isOn;

            if (_webCamTextureToMatHelper != null && _webCamTextureToMatHelper.IsInitialized())
                _webCamTextureToMatHelper.Initialize();
        }

        /// <summary>
        /// Raises the dictionary id dropdown value changed event.
        /// </summary>
        public void OnDictionaryIdDropdownValueChanged(int result)
        {
            if ((int)DictionaryId != result)
            {
                DictionaryId = (ArUcoDictionary)result;
                _dictionary = Objdetect.getPredefinedDictionary((int)DictionaryId);

                if (_webCamTextureToMatHelper != null && _webCamTextureToMatHelper.IsInitialized())
                    _webCamTextureToMatHelper.Initialize();
            }
        }

        /// <summary>
        /// Raises the enable low pass filter toggle value changed event.
        /// </summary>
        public void OnEnableLowPassFilterToggleValueChanged()
        {
            EnableLowPassFilter = EnableLowPassFilterToggle.isOn;

            if (ArHelper != null)
            {
                foreach (ARGameObject arGameObject in ArHelper.ARGameObjects)
                {
                    if (arGameObject != null)
                        arGameObject.UseLowPassFilter = EnableLowPassFilter;
                }
            }
        }

        /// <summary>
        /// Raises the enable smoothing filter toggle value changed event.
        /// </summary>
        public void OnEnableSmoothingFilterToggleValueChanged()
        {
            EnableSmoothingFilter = EnableSmoothingFilterToggle.isOn;

            if (ArHelper != null)
            {
                foreach (ARGameObject arGameObject in ArHelper.ARGameObjects)
                {
                    if (arGameObject != null)
                        arGameObject.UseSmoothingFilter = EnableSmoothingFilter;
                }
            }
        }

        /// <summary>
        /// Raises the enable SOLVEPNP_ITERATIVE toggle value changed event.
        /// </summary>
        public void OnEnableSOLVEPNP_ITERATIVEToggleValueChanged()
        {
            EnableSOLVEPNP_ITERATIVE = EnableSOLVEPNP_ITERATIVEToggle.isOn;

            if (ArHelper != null)
            {
                foreach (ARGameObject arGameObject in ArHelper.ARGameObjects)
                {
                    if (arGameObject != null)
                        arGameObject.UseSOLVEPNP_ITERATIVE = EnableSOLVEPNP_ITERATIVE;
                }
            }
        }

        /// <summary>
        /// Called when an ARGameObject enters the ARCamera viewport.
        /// </summary>
        /// <param name="aRHelper"></param>
        /// <param name="arCamera"></param>
        /// <param name="arGameObject"></param>
        public void OnEnterARCameraViewport(ARHelper aRHelper, ARCamera arCamera, ARGameObject arGameObject)
        {
            Debug.Log("OnEnterARCamera arCamera.name " + arCamera.name + " arGameObject.name " + arGameObject.name);

            StartCoroutine(arGameObject.GetComponent<ARCube>().EnterAnimation(arGameObject.gameObject, 0f, 1f, 0.5f));
        }

        /// <summary>
        /// Called when an ARGameObject exits the ARCamera viewport.
        /// </summary>
        /// <param name="aRHelper"></param>
        /// <param name="arCamera"></param>
        /// <param name="arGameObject"></param>
        public void OnExitARCameraViewport(ARHelper aRHelper, ARCamera arCamera, ARGameObject arGameObject)
        {
            Debug.Log("OnExitARCamera arCamera.name " + arCamera.name + " arGameObject.name " + arGameObject.name);

            StartCoroutine(arGameObject.GetComponent<ARCube>().ExitAnimation(arGameObject.gameObject, 1f, 0f, 0.2f));
        }

        private Mat CreateCameraMatrix(double fx, double fy, double cx, double cy)
        {
            Mat camMatrix = new Mat(3, 3, CvType.CV_64FC1);
            camMatrix.put(0, 0, fx);
            camMatrix.put(0, 1, 0);
            camMatrix.put(0, 2, cx);
            camMatrix.put(1, 0, 0);
            camMatrix.put(1, 1, fy);
            camMatrix.put(1, 2, cy);
            camMatrix.put(2, 0, 0);
            camMatrix.put(2, 1, 0);
            camMatrix.put(2, 2, 1.0f);

            return camMatrix;
        }

        private void StartThread(Action action)
        {
            ThreadPool.QueueUserWorkItem(_ => action());
        }

        private void StopThread()
        {
            if (!IsThreadRunning)
                return;

            while (IsThreadRunning)
            {
                //Wait threading stop
            }
        }

        private void ThreadWorker()
        {
            IsThreadRunning = true;

            DetectARUcoMarker();

            lock (_executeOnMainThread)
            {
                if (_executeOnMainThread.Count == 0)
                {
                    _executeOnMainThread.Enqueue(() =>
                    {
                        OnDetectionDone();
                    });
                }
            }

            IsThreadRunning = false;
        }

        private void DetectARUcoMarker()
        {
            // Get thread-safe copy of downScaleMat (already copied in Update())
            List<Mat> corners = new List<Mat>();
            Mat ids = new Mat();
            List<Mat> rejectedCorners = new List<Mat>();

            try
            {
                // Check if _downScaleMatForWorker is available (worker thread is exclusive, so safe to access)
                lock (_sync)
                {
                    if (_downScaleMatForWorker == null || _downScaleMatForWorker.empty())
                    {
                        lock (_sync)
                        {
                            _detectionResults = new List<DetectionResult>();
                        }
                        return;
                    }
                }

                // Detect markers using _downScaleMatForWorker and _undistortedRgbMatForWorker (already thread-safe copies from main thread)
                Calib3d.undistort(_downScaleMatForWorker, _undistortedRgbMatForWorker, _camMatrixForWorker, _distCoeffsForWorker);
                _arucoDetector.detectMarkers(_undistortedRgbMatForWorker, corners, ids, rejectedCorners);

                // Estimate pose if markers detected
                if (ApplyEstimationPose && ids.total() > 0)
                {
                    EstimatePoseCanonicalMarker(_undistortedRgbMatForWorker, corners, ids);
                }
                else
                {
                    // Store empty results for main thread processing
                    lock (_sync)
                    {
                        _detectionResults = new List<DetectionResult>();
                    }
                }
            }
            finally
            {
                // Clean up thread-local Mats
                ids?.Dispose();
                if (corners != null) foreach (var item in corners) item.Dispose();
                if (rejectedCorners != null) foreach (var item in rejectedCorners) item.Dispose();
            }
        }

        private void OnDetectionDone()
        {
            DebugUtils.TrackTick();

            // Reset ARGameObjects ImagePoints and ObjectPoints.
            if (ApplyEstimationPose && ArHelper != null)
            {
                ArHelper.ResetARGameObjectsImagePointsAndObjectPoints();

                // Apply detection results to ARGameObjects
                List<DetectionResult> detectionResults;
                lock (_sync)
                {
                    detectionResults = new List<DetectionResult>(_detectionResults);
                }

                foreach (var result in detectionResults)
                {
                    var arUcoId = new ArUcoIdentifier((int)_selectedMarkerType, (int)DictionaryId, new[] { result.MarkerId });
                    ARGameObject aRGameObject = FindOrCreateARGameObject(ArHelper.ARGameObjects, arUcoId, ArHelper.transform);

                    aRGameObject.ImagePoints = result.ImagePoints;
                    aRGameObject.ObjectPoints = result.ObjectPoints;
                }
            }

            if (DisplayCameraPreview)
            {
                Imgproc.cvtColor(_downScaleMat, _rgbMatForPreview, Imgproc.COLOR_GRAY2RGB);

                List<DetectionResult> detectionResults;
                lock (_sync)
                {
                    detectionResults = new List<DetectionResult>(_detectionResults);
                }
                foreach (var result in detectionResults)
                {
                    using (MatOfPoint2f imagePoints = new MatOfPoint2f(result.ImagePoints))
                    using (MatOfPoint3f objectPoints = new MatOfPoint3f(result.ObjectPoints))
                    {
                        DebugDrawFrameAxes(_rgbMatForPreview, objectPoints, imagePoints, _camMatrix, _distCoeffs, MarkerLength * 0.5f);
                    }
                }

                OpenCVMatUtils.MatToTexture2D(_rgbMatForPreview, _texture);
            }

            IsDetecting = false;
        }

        /// <summary>
        /// Finds or creates an ARGameObject with the specified AR marker identifier.
        /// </summary>
        /// <param name="arGameObjects"></param>
        /// <param name="arUcoId"></param>
        /// <param name="parentTransform"></param>
        /// <returns></returns>
        private ARGameObject FindOrCreateARGameObject(List<ARGameObject> arGameObjects, ArUcoIdentifier arUcoId, Transform parentTransform)
        {
            ARGameObject FindARGameObjectById(List<ARGameObject> arGameObjects, ArUcoIdentifier id)
            {
                if (_arGameObjectCache.TryGetValue(id, out var cachedObject))
                {
                    return cachedObject;
                }
                return null;
            }

            ARGameObject arGameObject = FindARGameObjectById(arGameObjects, arUcoId);
            if (arGameObject == null)
            {
                arGameObject = Instantiate(ArCubePrefab, parentTransform).GetComponent<ARGameObject>();

                string markerIdsStr = arUcoId.MarkerIds != null ? string.Join(",", arUcoId.MarkerIds) : null;
                string arUcoIdNameStr;
                if (markerIdsStr != null)
                    arUcoIdNameStr = (MarkerType)arUcoId.MarkerType + " " + (ArUcoDictionary)arUcoId.DictionaryId + " [" + markerIdsStr + "]";
                else
                    arUcoIdNameStr = (MarkerType)arUcoId.MarkerType + " " + (ArUcoDictionary)arUcoId.DictionaryId;

                arGameObject.name = arUcoIdNameStr;
                arGameObject.GetComponent<ARCube>().SetInfoPlateTexture(arUcoIdNameStr);
                arGameObject.UseLowPassFilter = EnableLowPassFilter;
                arGameObject.UseSmoothingFilter = EnableSmoothingFilter;
                arGameObject.UseSOLVEPNP_ITERATIVE = EnableSOLVEPNP_ITERATIVE;
                arGameObject.OnEnterARCameraViewport.AddListener(OnEnterARCameraViewport);
                arGameObject.OnExitARCameraViewport.AddListener(OnExitARCameraViewport);
                arGameObject.gameObject.SetActive(false);
                arGameObjects.Add(arGameObject);
                _arGameObjectCache[arUcoId] = arGameObject;
            }
            return arGameObject;
        }

        /// <summary>
        /// Removes all ARGameObjects from the list and destroys them.
        /// </summary>
        /// <param name="arGameObjects"></param>
        private void RemoveAllARGameObject(List<ARGameObject> arGameObjects)
        {
            foreach (ARGameObject arGameObject in arGameObjects)
            {
                Destroy(arGameObject.gameObject);
            }
            arGameObjects.Clear();

            _arGameObjectCache.Clear();
        }

        private void DebugDrawFrameAxes(Mat image, MatOfPoint3f objectPoints, MatOfPoint2f imagePoints, Mat cameraMatrix, MatOfDouble distCoeffs,
                                 float length, int thickness = 3)
        {
            // Calculate rvec and tvec for debug display and draw with Calib3d.drawFrameAxes()
            using (Mat rvec = new Mat(3, 1, CvType.CV_64FC1))
            using (Mat tvec = new Mat(3, 1, CvType.CV_64FC1))
            {
                // Calculate pose
                Calib3d.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

                // In this example we are processing with RGB color image, so Axis-color correspondences are X: blue, Y: green, Z: red. (Usually X: red, Y: green, Z: blue)
                OpenCVARUtils.SafeDrawFrameAxes(image, cameraMatrix, distCoeffs, rvec, tvec, length, thickness);
            }
        }

        private struct ArUcoIdentifier : IEquatable<ArUcoIdentifier>
        {
            public int MarkerType;    // enum value
            public int DictionaryId;  // enum value
            public int[] MarkerIds;   // marker ID array

            public ArUcoIdentifier(int markerType, int dictionaryId, int[] markerIds)
            {
                MarkerType = markerType;
                DictionaryId = dictionaryId;
                MarkerIds = markerIds;
            }

            public override string ToString()
            {
                string markerIdsStr = MarkerIds != null ? string.Join(",", MarkerIds) : null;
                if (markerIdsStr != null)
                    return $"{MarkerType} {DictionaryId} [{markerIdsStr}]";
                else
                    return $"{MarkerType} {DictionaryId}";
            }

            public override int GetHashCode()
            {
                // fast hash calculation
                int hash = MarkerType;
                hash = hash * 31 + DictionaryId;
                if (MarkerIds != null)
                {
                    foreach (int id in MarkerIds)
                    {
                        hash = hash * 31 + id;
                    }
                }
                return hash;
            }

            public bool Equals(ArUcoIdentifier other)
            {
                if (MarkerType != other.MarkerType || DictionaryId != other.DictionaryId)
                    return false;

                if (MarkerIds == null) return other.MarkerIds == null;
                if (other.MarkerIds == null) return false;

                if (MarkerIds.Length != other.MarkerIds.Length)
                    return false;

                for (int i = 0; i < MarkerIds.Length; i++)
                {
                    if (MarkerIds[i] != other.MarkerIds[i])
                        return false;
                }
                return true;
            }
        }

        private void EstimatePoseCanonicalMarker(Mat rgbMat, List<Mat> corners, Mat ids)
        {
            using (MatOfPoint3f objectPoints = new MatOfPoint3f(
                new Point3(-MarkerLength / 2f, MarkerLength / 2f, 0),
                new Point3(MarkerLength / 2f, MarkerLength / 2f, 0),
                new Point3(MarkerLength / 2f, -MarkerLength / 2f, 0),
                new Point3(-MarkerLength / 2f, -MarkerLength / 2f, 0)
                ))
            {
                // Store detection results for thread-safe transfer to main thread
                List<DetectionResult> detectionResults = new List<DetectionResult>();

#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                Span<int> idsValues = ids.AsSpan<int>();
#else
                int[] idsArray = new int[ids.total() * ids.channels()];
                ids.get(0, 0, idsArray);
                int[] idsValues = idsArray;
#endif

                for (int i = 0; i < idsValues.Length; i++)
                {
                    using (Mat corner_4x1 = corners[i].reshape(2, 4)) // 1*4*CV_32FC2 => 4*1*CV_32FC2
                    using (MatOfPoint2f imagePoints = new MatOfPoint2f(corner_4x1))
                    {
                        // Convert to thread-safe data structures
                        DetectionResult result = new DetectionResult
                        {
                            MarkerId = idsValues[i],
                            ImagePoints = imagePoints.toVector2Array(),
                            ObjectPoints = objectPoints.toVector3Array()
                        };
                        detectionResults.Add(result);
                    }
                }

                // Store results for main thread processing
                lock (_sync)
                {
                    _detectionResults = detectionResults;
                }
            }
        }
    }
}
