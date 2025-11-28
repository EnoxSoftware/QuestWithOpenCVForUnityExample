#if !UNITY_WSA_10_0

using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Oculus.Interaction;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace QuestWithOpenCVForUnityExample
{
    /// <summary>
    /// Quest Face Identification Estimator Example
    /// An example of using OpenCV dnn module with Face Detection and Recognition on MetaQuest.
    /// This example demonstrates face detection, face registration, and face identification.
    ///
    /// [Tested Models]
    /// Face Detection: face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
    /// Face Recognition: face_recognition_sface_2021dec.onnx https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
    /// </summary>
    [RequireComponent(typeof(QuestPassthrough2MatHelper))]
    public class QuestFaceIdentificationEstimatorExample : MonoBehaviour
    {
        // Public Fields
        [Header("UI")]
        public Toggle UseAsyncInferenceToggle;
        public bool UseAsyncInference = false;
        public Toggle ShowPassthroughImageToggle;
        public bool ShowPassthroughImage = false;

        [Header("Model Settings")]
        [Tooltip("Path to a binary file of face detection model contains trained weights.")]
        public string FaceDetectionModel = "OpenCVForUnityExamples/objdetect/face_detection_yunet_2023mar.onnx";

        [Tooltip("Path to a binary file of face recognition model contains trained weights.")]
        public string FaceRecognitionModel = "OpenCVForUnityExamples/objdetect/face_recognition_sface_2021dec.onnx";

        [Tooltip("Path to a text file of model contains network configuration.")]
        public string Config;

        [Tooltip("Confidence threshold.")]
        public float ConfThreshold = 0.6f;

        [Tooltip("Non-maximum suppression threshold.")]
        public float NmsThreshold = 0.3f;

        [Tooltip("Maximum detections per image.")]
        public int TopK = 100;

        [Tooltip("Preprocess input image by resizing to a specific width.")]
        public int InpWidth = 320;

        [Tooltip("Preprocess input image by resizing to a specific height.")]
        public int InpHeight = 320;

        [Header("Face Registration")]
        [Tooltip("Input field for face name registration.")]
        public InputField FaceNameInput;

        [Tooltip("Button to clear all registered faces.")]
        public Button ClearFacesButton;

        [Tooltip("Layout Group for displaying registered faces.")]
        public UnityEngine.UI.LayoutGroup RegisteredFacesLayoutGroup;

        [Space(10)]

        [HeaderAttribute("Debug")]

        public Text RenderFPS;
        public Text VideoFPS;
        public Text TrackFPS;
        public Text DebugStr;

        // Private Fields
        private Texture2D _texture;
        private Renderer _quadRenderer;
        private QuestPassthrough2MatHelper _webCamTextureToMatHelper;
        private Mat _bgrMat;
        private FaceIdentificationEstimator _faceIdentificationEstimator;
        private string _configFilepath;
        private string _faceDetectionModelFilepath;
        private string _faceRecognitionModelFilepath;
        private CancellationTokenSource _cts = new CancellationTokenSource();
        private Mat _bgrMatForAsync;
        private Mat _latestDetectedFaces;
        private Task _inferenceTask;
        private readonly Queue<Action> _mainThreadQueue = new();
        private readonly object _queueLock = new();
        private bool _shouldUpdateFromPoint = false;
        private Vector2 _selectedUVPoint;
        private RayInteractor[] _rayInteractors;
        private Dictionary<RayInteractor, InteractorState> _previousStates;
        private readonly Dictionary<int, RawImage> _registeredFaceRawImages = new Dictionary<int, RawImage>();
        private readonly Dictionary<int, Texture2D> _registeredFaceTextures = new Dictionary<int, Texture2D>();

        // Unity Lifecycle Methods
        private void Awake()
        {
            // Find all RayInteractors in the scene (controllers, hand rays, etc.)
            _rayInteractors = FindObjectsByType<RayInteractor>(FindObjectsSortMode.None);
            if (_rayInteractors == null || _rayInteractors.Length == 0)
            {
                Debug.LogWarning("RayInteractor not found in the scene. Point selection will not work.");
            }
            else
            {
                Debug.Log($"Found {_rayInteractors.Length} RayInteractor(s) in the scene.");
            }

            // Initialize dictionary to track previous states of RayInteractors
            _previousStates = new Dictionary<RayInteractor, InteractorState>();
        }

        private async void Start()
        {
            _webCamTextureToMatHelper = gameObject.GetComponent<QuestPassthrough2MatHelper>();
            _webCamTextureToMatHelper.OutputColorFormat = Source2MatHelperColorFormat.RGBA;

            // Update GUI state
            UseAsyncInferenceToggle.isOn = UseAsyncInference;
            ShowPassthroughImageToggle.isOn = ShowPassthroughImage;

            // Asynchronously retrieves the readable file path from the StreamingAssets directory.
            DebugUtils.AddDebugStr("Preparing file access...");

            if (!string.IsNullOrEmpty(Config))
            {
                _configFilepath = await OpenCVEnv.GetFilePathTaskAsync(Config, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_configFilepath)) Debug.Log("The file:" + Config + " did not exist.");
            }
            if (!string.IsNullOrEmpty(FaceDetectionModel))
            {
                _faceDetectionModelFilepath = await OpenCVEnv.GetFilePathTaskAsync(FaceDetectionModel, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_faceDetectionModelFilepath)) Debug.Log("The file:" + FaceDetectionModel + " did not exist.");
            }
            if (!string.IsNullOrEmpty(FaceRecognitionModel))
            {
                _faceRecognitionModelFilepath = await OpenCVEnv.GetFilePathTaskAsync(FaceRecognitionModel, cancellationToken: _cts.Token);
                if (string.IsNullOrEmpty(_faceRecognitionModelFilepath)) Debug.Log("The file:" + FaceRecognitionModel + " did not exist.");
            }

            DebugUtils.ClearDebugStr();

            Run();
        }

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
#endif

            float halfOfVerticalFov = Mathf.Atan(1.0f / projectionMatrix.m11);
            float aspectRatio = (1.0f / Mathf.Tan(halfOfVerticalFov)) / projectionMatrix.m00;
            Debug.Log("halfOfVerticalFov " + halfOfVerticalFov);
            Debug.Log("aspectRatio " + aspectRatio);

            // Calculate Quad scale for 2.2m distance
            float distance = 2.2f;
            float quadHeight = 2.0f * distance * Mathf.Tan(halfOfVerticalFov);
            float quadWidth = quadHeight * aspectRatio;
            gameObject.transform.localScale = new Vector3(quadWidth, quadHeight, 1.0f);

            _bgrMat = new Mat(rgbaMat.rows(), rgbaMat.cols(), CvType.CV_8UC3);
            _bgrMatForAsync = new Mat();

            DebugUtils.AddDebugStr("Touch a detected face to register it.");
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
            _latestDetectedFaces?.Dispose(); _latestDetectedFaces = null;

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

            // Check all RayInteractors for selection state transition
            if (_rayInteractors != null && _rayInteractors.Length > 0)
            {
                foreach (var interactor in _rayInteractors)
                {
                    if (interactor == null)
                        continue;

                    InteractorState currentState = interactor.State;

                    // Check if previous state exists
                    if (_previousStates.TryGetValue(interactor, out InteractorState previousState))
                    {
                        // Detect transition from Select to non-Select state (click release)
                        if (previousState == InteractorState.Select && currentState != InteractorState.Select)
                        {
                            // Skip processing if the RayInteractor is pointing at UI elements
                            if (!IsPointingAtUI(interactor))
                            {
                                ProcessSelection(interactor);
                            }
                        }
                    }

                    // Update the previous state for next frame
                    _previousStates[interactor] = currentState;
                }
            }

            if (_webCamTextureToMatHelper.IsPlaying() && _webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                DebugUtils.VideoTick();

                Mat rgbaMat = _webCamTextureToMatHelper.GetMat();

                if (_faceIdentificationEstimator == null)
                {
                    Imgproc.putText(rgbaMat, "model files are not loaded.", new Point(5, rgbaMat.rows() - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255), 2, Imgproc.LINE_AA, false);
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
                                    // Face identification inference
                                    var newFaces = await _faceIdentificationEstimator.EstimateAsync(_bgrMatForAsync);
                                    RunOnMainThread(() =>
                                        {
                                            _latestDetectedFaces?.Dispose();
                                            _latestDetectedFaces = newFaces;
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

                        if (_latestDetectedFaces != null)
                        {
                            _faceIdentificationEstimator.Visualize(rgbaMat, _latestDetectedFaces, false, true);

                            // Check for point selection completion and register face
                            if (_shouldUpdateFromPoint)
                            {
                                // Convert UV coordinate to OpenCV Point
                                Point selectedPoint = ConvertUVToOpenCVPoint(_selectedUVPoint, rgbaMat.cols(), rgbaMat.rows());
                                RegisterSelectedFace(_bgrMat, _latestDetectedFaces, selectedPoint);

                                // Update face recognition for all tracked faces with the new registered face
                                _faceIdentificationEstimator.UpdateFaceRecognitionForAllTrackedFaces(_bgrMat, true);

                                _shouldUpdateFromPoint = false;
                            }
                        }
                    }
                    else
                    {
                        // synchronous execution

                        // Face identification inference
                        using (Mat faces = _faceIdentificationEstimator.Estimate(_bgrMat))
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

                            _faceIdentificationEstimator.Visualize(rgbaMat, faces, false, true);

                            // Check for point selection completion and register face
                            if (_shouldUpdateFromPoint)
                            {
                                // Convert UV coordinate to OpenCV Point
                                Point selectedPoint = ConvertUVToOpenCVPoint(_selectedUVPoint, rgbaMat.cols(), rgbaMat.rows());
                                RegisterSelectedFace(_bgrMat, faces, selectedPoint);

                                // Update face recognition for all tracked faces with the new registered face
                                _faceIdentificationEstimator.UpdateFaceRecognitionForAllTrackedFaces(_bgrMat, true);

                                _shouldUpdateFromPoint = false;
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

            _faceIdentificationEstimator?.Dispose();

            // Clear all DebugMat windows on destroy
            DebugMat.destroyAllWindows();

            // Clear all registered face UGUI elements
            ClearRegisteredFaceUI();

            OpenCVDebug.SetDebugMode(false);

            _cts?.Dispose();
        }

        // Public Methods
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
        /// Waits for any ongoing inference to complete before changing the toggle state.
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
        /// Clears all registered faces, resets face recognition for all tracked faces, and clears all UGUI RawImages and textures.
        /// </summary>
        public void OnClearFacesButtonClick()
        {
            if (_faceIdentificationEstimator != null)
            {
                _faceIdentificationEstimator.ClearRegisteredFaces();
                Debug.Log("All registered faces cleared.");

                _faceIdentificationEstimator.ResetFaceRecognitionForAllTrackedFaces();
                Debug.Log("Face recognition reset for all tracked faces.");
            }

            // Clear all UGUI RawImages and textures
            ClearRegisteredFaceUI();
        }

        /// <summary>
        /// Clears all registered face UGUI RawImages and textures.
        /// </summary>
        private void ClearRegisteredFaceUI()
        {
            // Destroy all RawImage GameObjects
            foreach (var kvp in _registeredFaceRawImages)
            {
                if (kvp.Value != null)
                {
                    UnityEngine.Object.Destroy(kvp.Value.gameObject);
                }
            }
            _registeredFaceRawImages.Clear();

            // Destroy all textures
            foreach (var kvp in _registeredFaceTextures)
            {
                if (kvp.Value != null)
                {
                    Texture2D.Destroy(kvp.Value);
                }
            }
            _registeredFaceTextures.Clear();

            Debug.Log("All registered face UGUI elements cleared.");
        }

        /// <summary>
        /// Checks if the RayInteractor is pointing at a UI element.
        /// </summary>
        /// <param name="rayInteractor">The RayInteractor to check.</param>
        /// <returns>True if the RayInteractor is pointing at a UI element, false otherwise.</returns>
        private bool IsPointingAtUI(RayInteractor rayInteractor)
        {
            if (rayInteractor == null)
                return false;

            // Check if RayInteractor has a candidate (something it's pointing at)
            if (!rayInteractor.HasCandidate)
                return false;

            var candidate = rayInteractor.Candidate;

            // Check if the candidate is a UI element
            // Check if the candidate's GameObject has a Canvas component or is on the UI layer
            if (candidate != null)
            {
                GameObject candidateGameObject = candidate.gameObject;

                // Check if the GameObject has a Canvas component (UGUI)
                if (candidateGameObject.GetComponent<Canvas>() != null ||
                    candidateGameObject.GetComponentInParent<Canvas>() != null)
                {
                    return true;
                }

                // Check if the GameObject is on the UI layer (layer 5)
                if (candidateGameObject.layer == LayerMask.NameToLayer("UI"))
                {
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        /// Processes the selection from a RayInteractor.
        /// Uses Physics.Raycast to get UV coordinates since direct access from RayInteractor is not available.
        /// </summary>
        /// <param name="rayInteractor">The RayInteractor that triggered the selection.</param>
        private void ProcessSelection(RayInteractor rayInteractor)
        {
            if (rayInteractor == null)
                return;

            // Get actual ray from RayInteractor (derived from Origin and Forward)
            UnityEngine.Ray ray = rayInteractor.Ray;
            Vector3 origin = ray.origin;
            Vector3 direction = ray.direction;

            // Perform Physics.Raycast to get RaycastHit with UV coordinates
            // Use a reasonable max distance (10 meters)
            RaycastHit hit;
            if (Physics.Raycast(origin, direction, out hit, 10f))
            {
                // Check if the hit collider belongs to this quad
                if (hit.collider != null && hit.collider.gameObject == gameObject)
                {
                    Vector2 uv = hit.textureCoord;
                    _selectedUVPoint = uv;
                    _shouldUpdateFromPoint = true;
                    Debug.Log($"Face point selected at UV: ({uv.x}, {uv.y}) from {rayInteractor.name}");
                }
            }
        }

        // Private Methods
        /// <summary>
        /// Converts Unity UV coordinate (bottom-left origin) to OpenCV Point (top-left origin).
        /// </summary>
        /// <param name="uv">Unity UV coordinate (0-1 range, bottom-left origin).</param>
        /// <param name="width">Image width.</param>
        /// <param name="height">Image height.</param>
        /// <returns>OpenCV Point (top-left origin).</returns>
        private Point ConvertUVToOpenCVPoint(Vector2 uv, int width, int height)
        {
            // Unity UV: (0,0) is bottom-left, (1,1) is top-right
            // OpenCV: (0,0) is top-left, (width, height) is bottom-right
            double x = uv.x * width;
            double y = (1.0 - uv.y) * height;
            return new Point(x, y);
        }

        /// <summary>
        /// Registers the face that was selected by point selection.
        /// If the selected face already has a registered face ID, it updates the registration only if the current confidence is higher.
        /// If the selected face is new, it creates a new registration with the name from FaceNameInput or a default name.
        /// After registration, it displays the registered face using DebugMat.
        /// </summary>
        /// <param name="image">The input image containing the faces.</param>
        /// <param name="detectedFaces">The detected faces matrix.</param>
        /// <param name="selectedPoint">The selected point coordinates.</param>
        private void RegisterSelectedFace(Mat image, Mat detectedFaces, Point selectedPoint)
        {
            if (_faceIdentificationEstimator == null || detectedFaces == null || detectedFaces.empty())
            {
                Debug.LogWarning("No face detection estimator or no faces detected.");
                return;
            }

            if (image == null)
            {
                Debug.LogWarning("Input image is null.");
                return;
            }

            // Convert detection results to structured data for efficient access
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
            Span<FaceIdentificationEstimator.FaceIdentificationData> facesData = _faceIdentificationEstimator.ToStructuredDataAsSpan(detectedFaces);
#else
            FaceIdentificationEstimator.FaceIdentificationData[] facesData = _faceIdentificationEstimator.ToStructuredData(detectedFaces);
#endif

            // Find the face containing the selected point
            int bestFaceIndex = FindFaceContainingPoint(facesData, selectedPoint);

            if (bestFaceIndex >= 0)
            {
                // Get the selected face data
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                ref readonly var selectedFaceData = ref facesData[bestFaceIndex];
#else
                var selectedFaceData = facesData[bestFaceIndex];
#endif

                // Check if this face already has a faceId
                int existingFaceId = (int)selectedFaceData.FaceId;
                Debug.Log($"Selected face ID: {existingFaceId}");
                float currentConfidence = selectedFaceData.Score;

                // Create a face row for alignment using the conversion method
                Mat faceRow = FaceIdentificationEstimator.ConvertFaceDetectionDataToMat(selectedFaceData.FaceDetection);

                int faceId;
                string faceName;

                if (existingFaceId >= 0)
                {
                    // Face is already recognized - use existing face name
                    faceId = existingFaceId;

                    // Get existing face name
                    string existingFaceName = _faceIdentificationEstimator.GetFaceName(existingFaceId);
                    faceName = existingFaceName ?? $"Face_{existingFaceId}";

                    float existingConfidence = _faceIdentificationEstimator.GetFaceDetectionConfidence(existingFaceId);

                    Debug.Log($"Selected face is already registered with ID: {existingFaceId}, existing confidence: {existingConfidence:F3}, current confidence: {currentConfidence:F3}");

                    if (currentConfidence > existingConfidence)
                    {
                        Debug.Log($"Updating face ID {existingFaceId} with higher confidence: {existingConfidence:F3} -> {currentConfidence:F3}");
                        _faceIdentificationEstimator.RegisterFaceFromDetection(image, faceRow, faceId, faceName);
                        //
                        DebugUtils.AddDebugStr($"Updating face ID {existingFaceId} with higher confidence: {existingConfidence:F3} -> {currentConfidence:F3}");
                        //
                    }
                    else
                    {
                        Debug.Log($"Face ID {existingFaceId} already has higher or equal confidence: {existingConfidence:F3} >= {currentConfidence:F3}, skipping update");
                        //
                        DebugUtils.AddDebugStr($"Face ID {existingFaceId} already has higher or equal confidence: {existingConfidence:F3} >= {currentConfidence:F3}, skipping update");
                        //
                    }
                }
                else
                {
                    // New face registration - generate new face name
                    faceId = _faceIdentificationEstimator.RegisteredFaceCount + 1;

                    if (FaceNameInput != null && !string.IsNullOrEmpty(FaceNameInput.text?.Trim()))
                    {
                        faceName = FaceNameInput.text.Trim();
                    }
                    else
                    {
                        faceName = $"Face_{faceId}";
                    }

                    _faceIdentificationEstimator.RegisterFaceFromDetection(image, faceRow, faceId, faceName);
                    Debug.Log($"Face registered successfully: {faceName} (ID: {faceId})");
                    //
                    DebugUtils.AddDebugStr($"Face registered successfully: {faceName} (ID: {faceId})");
                    //
                }

                faceRow.Dispose();

                // Display the registered face using UGUI
                DisplayRegisteredFaceUGUI(faceId);
            }
            else
            {
                Debug.LogWarning("No face found near the selected point.");
                //
                DebugUtils.AddDebugStr($"No face found near the selected point.");
                //
            }
        }

        /// <summary>
        /// Displays the registered face using UGUI RawImage with annotations including face ID, name, confidence score, and colored border.
        /// </summary>
        /// <param name="faceId">The face ID.</param>
        private void DisplayRegisteredFaceUGUI(int faceId)
        {
            if (_faceIdentificationEstimator == null)
                return;

            if (RegisteredFacesLayoutGroup == null)
            {
                Debug.LogWarning("RegisteredFacesLayoutGroup is not set. Cannot display registered face.");
                return;
            }

            Mat alignedFace = _faceIdentificationEstimator.GetAlignedFace(faceId);
            if (alignedFace == null || alignedFace.empty())
                return;

            // Get face name
            string faceName = _faceIdentificationEstimator.GetFaceName(faceId);
            if (faceName == null)
                faceName = $"Face_{faceId}";

            // Create a copy for drawing text
            Mat displayFace = alignedFace.clone();

            // Get image dimensions for proper text positioning (BGR mat)
            int imgWidth = displayFace.cols();
            int imgHeight = displayFace.rows();

            // Prepare text color and draw border around the entire Mat using it
            Scalar textColor = _faceIdentificationEstimator.GetColorForFaceId(faceId);
            Imgproc.rectangle(displayFace, new Point(0, 0), new Point(imgWidth - 1, imgHeight - 1), textColor, 2);

            // Draw face ID and name on the image
            string displayText = $"ID: {faceId} ({faceName})";

            // Calculate font scale to fit text within image width
            double fontScale = 0.5;
            int thickness = 1;

            // Get text size to check if it fits
            Size textSize = Imgproc.getTextSize(displayText, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, null);

            // Adjust font scale if text is too wide
            if (textSize.width > imgWidth - 10)
            {
                fontScale = (imgWidth - 10) / (double)textSize.width * fontScale;
            }

            // Draw label inside a filled rectangle attached to top-left of the Mat
            int[] baseLineTop = new int[1];
            var labelSizeTop = Imgproc.getTextSizeAsValueTuple(displayText, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseLineTop);
            double rectLeftTop = 0d;
            double rectTopTop = 0d;
            Imgproc.rectangle(displayFace,
                new Point(rectLeftTop, rectTopTop),
                new Point(rectLeftTop + labelSizeTop.width, rectTopTop + labelSizeTop.height + baseLineTop[0]),
                textColor, Core.FILLED);
            Imgproc.putText(displayFace, displayText, new Point(rectLeftTop, rectTopTop + labelSizeTop.height), Imgproc.FONT_HERSHEY_SIMPLEX, fontScale, new Scalar(255, 255, 255, 255), thickness, Imgproc.LINE_AA, false);

            // Draw confidence score at bottom-left
            float confidence = _faceIdentificationEstimator.GetFaceDetectionConfidence(faceId);
            string confidenceText = $"Confidence: {confidence:F3}";
            Scalar confidenceColor = _faceIdentificationEstimator.GetColorForFaceId(faceId);

            // Calculate font scale for confidence text
            double confidenceFontScale = 0.4;
            int confidenceThickness = 1;

            Size confidenceTextSize = Imgproc.getTextSize(confidenceText, Imgproc.FONT_HERSHEY_SIMPLEX, confidenceFontScale, confidenceThickness, null);

            // Adjust font scale if confidence text is too wide
            if (confidenceTextSize.width > imgWidth - 10)
            {
                confidenceFontScale = (imgWidth - 10) / (double)confidenceTextSize.width * confidenceFontScale;
            }

            // Draw confidence inside a filled rectangle attached to bottom-right of the Mat
            int[] baseLineBottom = new int[1];
            var labelSizeBottom = Imgproc.getTextSizeAsValueTuple(confidenceText, Imgproc.FONT_HERSHEY_SIMPLEX, confidenceFontScale, confidenceThickness, baseLineBottom);
            double rectRight = imgWidth;
            double rectBottom = imgHeight;
            double rectLeftBottom = rectRight - labelSizeBottom.width;
            double rectTopBottom = rectBottom - (labelSizeBottom.height + baseLineBottom[0]);
            Imgproc.rectangle(displayFace,
                new Point(rectLeftBottom, rectTopBottom),
                new Point(rectRight, rectBottom),
                confidenceColor, Core.FILLED);
            Imgproc.putText(displayFace, confidenceText, new Point(rectLeftBottom, rectBottom - baseLineBottom[0]), Imgproc.FONT_HERSHEY_SIMPLEX, confidenceFontScale, new Scalar(255, 255, 255, 255), confidenceThickness, Imgproc.LINE_AA, false);

            // Convert BGR to RGBA for Texture2D
            Mat rgbaFace = new Mat();
            Imgproc.cvtColor(displayFace, rgbaFace, Imgproc.COLOR_BGR2RGBA);

            // Get or create RawImage for this faceId
            RawImage rawImage = null;
            if (_registeredFaceRawImages.TryGetValue(faceId, out rawImage))
            {
                // Check if the RawImage was destroyed (Unity's null check)
                if (rawImage == null)
                {
                    // RawImage was destroyed, remove from dictionary and clean up texture
                    _registeredFaceRawImages.Remove(faceId);
                    if (_registeredFaceTextures.TryGetValue(faceId, out Texture2D oldTexture))
                    {
                        if (oldTexture != null) Texture2D.Destroy(oldTexture);
                        _registeredFaceTextures.Remove(faceId);
                    }
                    rawImage = null; // Ensure it's null for the check below
                }
                // If rawImage is not null here, it means we have a valid existing RawImage to update
            }

            if (rawImage == null)
            {
                // Create new RawImage GameObject
                GameObject rawImageObject = new GameObject($"RegisteredFace_{faceId}");
                rawImageObject.transform.SetParent(RegisteredFacesLayoutGroup.transform, false);
                rawImage = rawImageObject.AddComponent<RawImage>();

                // Set size: width 100px, maintain aspect ratio
                RectTransform rectTransform = rawImage.GetComponent<RectTransform>();
                float aspectRatio = (float)imgHeight / imgWidth;
                rectTransform.sizeDelta = new Vector2(100f, 100f * aspectRatio);

                _registeredFaceRawImages[faceId] = rawImage;
            }

            // Create or update Texture2D
            Texture2D texture;
            if (_registeredFaceTextures.TryGetValue(faceId, out texture))
            {
                if (texture != null)
                {
                    // Update existing texture if size matches
                    if (texture.width == rgbaFace.cols() && texture.height == rgbaFace.rows())
                    {
                        OpenCVMatUtils.MatToTexture2D(rgbaFace, texture);
                    }
                    else
                    {
                        // Size changed, recreate texture
                        Texture2D.Destroy(texture);
                        texture = new Texture2D(rgbaFace.cols(), rgbaFace.rows(), TextureFormat.RGBA32, false);
                        OpenCVMatUtils.MatToTexture2D(rgbaFace, texture);
                        _registeredFaceTextures[faceId] = texture;
                    }
                }
                else
                {
                    // Texture was destroyed, create new one
                    texture = new Texture2D(rgbaFace.cols(), rgbaFace.rows(), TextureFormat.RGBA32, false);
                    OpenCVMatUtils.MatToTexture2D(rgbaFace, texture);
                    _registeredFaceTextures[faceId] = texture;
                }
            }
            else
            {
                // Create new texture
                texture = new Texture2D(rgbaFace.cols(), rgbaFace.rows(), TextureFormat.RGBA32, false);
                OpenCVMatUtils.MatToTexture2D(rgbaFace, texture);
                _registeredFaceTextures[faceId] = texture;
            }

            // Set texture to RawImage
            rawImage.texture = texture;

            // Cleanup
            displayFace.Dispose();
            rgbaFace.Dispose();
            alignedFace.Dispose();
        }


        /// <summary>
        /// Finds the face that contains the selected point within its bounding box.
        /// </summary>
        /// <param name="facesData">The detected faces structured data.</param>
        /// <param name="selectedPoint">The selected point.</param>
        /// <returns>The index of the face containing the point, or -1 if no face contains the point.</returns>
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
        private int FindFaceContainingPoint(Span<FaceIdentificationEstimator.FaceIdentificationData> facesData, Point selectedPoint)
#else
        private int FindFaceContainingPoint(FaceIdentificationEstimator.FaceIdentificationData[] facesData, Point selectedPoint)
#endif
        {
            if (facesData == null || facesData.Length == 0)
                return -1;

            for (int i = 0; i < facesData.Length; i++)
            {
#if NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE
                ref readonly var faceData = ref facesData[i];
#else
                var faceData = facesData[i];
#endif

                // Extract bounding box coordinates (x, y, width, height)
                float x = faceData.X;
                float y = faceData.Y;
                float width = faceData.Width;
                float height = faceData.Height;

                // Check if the selected point is within the face bounding box
                if (selectedPoint.x >= x && selectedPoint.x <= x + width &&
                    selectedPoint.y >= y && selectedPoint.y <= y + height)
                {
                    return i; // Return the first face that contains the point
                }
            }

            return -1; // No face contains the selected point
        }

        /// <summary>
        /// Initializes the face identification estimator with model files and starts the quest passthrough to mat helper.
        /// </summary>
        private void Run()
        {
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            OpenCVDebug.SetDebugMode(true);


            if (string.IsNullOrEmpty(_faceDetectionModelFilepath) || string.IsNullOrEmpty(_faceRecognitionModelFilepath))
            {
                Debug.LogError("model files are not loaded. Please use [Tools] > [OpenCV for Unity] > [Setup Tools] > [Example Assets Downloader] to download the asset files required for this example scene, and then move them to the \"Assets/StreamingAssets\" folder.");
            }
            else
            {
                _faceIdentificationEstimator = new FaceIdentificationEstimator(_faceDetectionModelFilepath, _faceRecognitionModelFilepath, new Size(InpWidth, InpHeight), ConfThreshold, NmsThreshold, TopK);
            }

            _webCamTextureToMatHelper.Initialize();
        }

        /// <summary>
        /// Enqueues an action to be executed on the main thread.
        /// </summary>
        /// <param name="action">The action to execute on the main thread.</param>
        private void RunOnMainThread(Action action)
        {
            if (action == null) return;

            lock (_queueLock)
            {
                _mainThreadQueue.Enqueue(action);
            }
        }

        /// <summary>
        /// Processes all actions queued for execution on the main thread.
        /// </summary>
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
