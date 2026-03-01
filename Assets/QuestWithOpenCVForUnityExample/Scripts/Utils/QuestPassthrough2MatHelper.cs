#pragma warning disable 0067
#pragma warning disable 0618

#if !OPENCV_DONT_USE_WEBCAMTEXTURE_API
#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using System;
using System.Collections;
using System.Threading;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.Rendering;
using Rect = UnityEngine.Rect;


#if (UNITY_ANDROID && !UNITY_EDITOR) && !DISABLE_QUESTPASSTHROUGH_API
using Meta.XR;
#endif

namespace QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat
{
    /// <summary>
    /// A helper component class for obtaining camera frames from Meta Quest passthrough camera via MRUK PassthroughCameraAccess and converting them to OpenCV <c>Mat</c> format in real-time.
    /// </summary>
    /// <remarks>
    /// The <c>QuestPassthrough2MatHelper</c> class captures video frames from the passthrough camera using the
    /// <c>PassthroughCameraAccess</c> component (Mixed Reality Utility Toolkit) and converts each frame to an OpenCV <c>Mat</c> object.
    /// Uses AsyncGPUReadback for efficient GPU-to-CPU transfer. This component handles camera orientation, rotation, and necessary transformations.
    ///
    /// <strong>Note:</strong> By setting outputColorFormat to RGBA, processing that does not include extra color conversion is performed.
    /// </remarks>
    /// <example>
    /// Attach this component to a GameObject (with PassthroughCameraAccess, or it will be auto-added) and call <c>GetMat()</c> to retrieve the latest camera frame in <c>Mat</c> format.
    /// </example>
    /// <example>
    /// <code>
    /// Supported resolutions:
    /// 320 x 240
    /// 640 x 360
    /// 640 x 480
    /// 720 x 480
    /// 720 x 576
    /// 800 x 600
    /// 1024 x 576
    /// 1280 x 720
    /// 1280 x 960
    /// 1280 x 1080
    /// 1280 x 1280
    /// </code>
    /// </example>
    public class QuestPassthrough2MatHelper : WebCamTexture2MatHelper
    {
        [SerializeField] public PassthroughCameraEye Eye = PassthroughCameraEye.Left;

#if UNITY_EDITOR
        private void Reset()
        {
            _requestedWidth = 1280;
            _requestedHeight = 1280;
            _requestedFPS = 60f;
        }
#endif

#if (UNITY_ANDROID && !UNITY_EDITOR) && !DISABLE_QUESTPASSTHROUGH_API

        public override float RequestedFPS
        {
            get { return _requestedFPS; }
            set
            {
                _requestedFPS = Mathf.Clamp(value, -1f, float.MaxValue);
                if (hasInitDone)
                    Initialize();
                else if (isInitWaiting)
                    Initialize(autoPlayAfterInitialize);
            }
        }

        /// <summary>
        /// PassthroughCameraAccess component (GetComponent / AddComponent).
        /// </summary>
        protected PassthroughCameraAccess passthroughCameraAccess;

        /// <summary>
        /// Whether the mat has been updated in this frame (sync with AsyncGPUReadback completion).
        /// </summary>
        protected bool didUpdateThisFrame = false;

        /// <summary>
        /// Whether to use AsyncGPUReadback for Texture to Mat conversion.
        /// When false, uses synchronous Texture2D/Texture2DToMatRaw path.
        /// </summary>
        protected bool useAsyncGPUReadback;

        /// <summary>
        /// Texture2D buffer for synchronous readback when useAsyncGPUReadback is false.
        /// </summary>
        protected Texture2D texture2DBuffer;

        /// <summary>
        /// Current AsyncGPUReadbackRequest for direct data access.
        /// </summary>
        protected AsyncGPUReadbackRequest asyncGPUReadbackRequestBuffer;

        /// <summary>
        /// Event to signal when frame data is ready for processing.
        /// </summary>
        private ManualResetEventSlim frameDataReadyEvent;

        /// <summary>
        /// The waitForEndOfFrameCoroutine.
        /// </summary>
        protected IEnumerator waitForEndOfFrameCoroutine;

        /// <summary>
        /// Cached raw values from PassthroughCameraAccess. Updated every frame alongside CallReadback when IsPlaying.
        /// Used when Pause/Stop to return last known values. Semantically aligned with GetMat() frame data.
        /// </summary>
        private Pose _cachedPose;
        private Vector2 _cachedFocalLength;
        private Vector2 _cachedPrincipalPoint;
        private Vector2Int _cachedSensorResolution;
        private Vector2Int _cachedCurrentResolution;
        private bool _hasCachedValues;

        // Update is called once per frame
        protected override void Update()
        {
            if (hasInitDone)
            {
                if (passthroughCameraAccess == null || !passthroughCameraAccess.IsPlaying) return;

                CallReadback();
                UpdateCacheFromPassthrough();
            }
        }

        protected virtual void CallReadback()
        {
            var texture = passthroughCameraAccess?.GetTexture();
            if (texture == null) return;

            if (useAsyncGPUReadback)
            {
                AsyncGPUReadback.Request(texture, 0, OnCompleteReadback);
            }
            else
            {
                OpenCVMatUtils.TextureToTexture2D(texture, texture2DBuffer);
                if (frameDataReadyEvent != null)
                    frameDataReadyEvent.Reset();
            }
        }

        /// <summary>
        /// Thread-safe callback for completed AsyncGPUReadbackRequest.
        /// </summary>
        protected virtual void OnCompleteReadback(AsyncGPUReadbackRequest request)
        {
            asyncGPUReadbackRequestBuffer = request;
            if (frameDataReadyEvent != null)
                frameDataReadyEvent.Reset();
        }

        protected virtual IEnumerator _WaitForFrameEndCoroutine()
        {
            while (true)
            {
                yield return new WaitForEndOfFrame();

                if (!frameDataReadyEvent.IsSet)
                {
                    if (!useAsyncGPUReadback)
                    {
                        OpenCVMatUtils.Texture2DToMatRaw(texture2DBuffer, baseMat);
                        Core.flip(baseMat, baseMat, 0);
                    }
                    else
                    {
                        if (asyncGPUReadbackRequestBuffer.hasError)
                        {
                            frameDataReadyEvent.Set();
                            continue;
                        }

#if !OPENCV_DONT_USE_UNSAFE_CODE
                        OpenCVMatUtils.CopyToMat(asyncGPUReadbackRequestBuffer.GetData<byte>(), baseMat);
#endif
                        Core.flip(baseMat, baseMat, 0);
                    }

                    frameDataReadyEvent.Set();
                    didUpdateThisFrame = true;
                }
                else
                {
                    didUpdateThisFrame = false;
                }
            }
        }

        /// <summary>
        /// Initialize this instance by coroutine.
        /// </summary>
        protected override IEnumerator _Initialize()
        {
            if (hasInitDone)
            {
                CancelWaitForEndOfFrameCoroutine();
                ReleaseResources();
                if (_onDisposed != null)
                    _onDisposed.Invoke();
            }

#if !OPENCV_DONT_USE_UNSAFE_CODE
            useAsyncGPUReadback = SystemInfo.supportsAsyncGPUReadback;
#else
            useAsyncGPUReadback = false;
#endif

            isInitWaiting = true;
            yield return null;

            passthroughCameraAccess = GetComponent<PassthroughCameraAccess>();
            if (passthroughCameraAccess == null)
                passthroughCameraAccess = gameObject.AddComponent<PassthroughCameraAccess>();

            // MaxFramerate and RequestedResolution can only be changed when enabled is false
            passthroughCameraAccess.enabled = false;

            passthroughCameraAccess.CameraPosition = Eye == PassthroughCameraEye.Left
                ? PassthroughCameraAccess.CameraPositionType.Left
                : PassthroughCameraAccess.CameraPositionType.Right;

            if (RequestedWidth > 0 && RequestedHeight > 0)
            {
                passthroughCameraAccess.RequestedResolution = new Vector2Int(RequestedWidth, RequestedHeight);
            }

            if (_requestedFPS > 0)
            {
                passthroughCameraAccess.MaxFramerate = (int)_requestedFPS;
            }

            passthroughCameraAccess.enabled = true;

            int initFrameCount = 0;

            while (!passthroughCameraAccess.IsPlaying)
            {
                if (initFrameCount > TimeoutFrameCount)
                {
                    passthroughCameraAccess.enabled = false;
                    isInitWaiting = false;
                    initCoroutine = null;
                    if (_onErrorOccurred != null)
                        _onErrorOccurred.Invoke(Source2MatHelperErrorCode.TIMEOUT, "PassthroughCameraAccess did not start.");
                    yield break;
                }
                initFrameCount++;
                yield return null;
            }

            var resolution = passthroughCameraAccess.CurrentResolution;
            int width = resolution.x;
            int height = resolution.y;

            if (width <= 0 || height <= 0)
            {
                passthroughCameraAccess.enabled = false;
                isInitWaiting = false;
                initCoroutine = null;
                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.UNKNOWN, "Invalid camera resolution.");
                yield break;
            }

            if (!useAsyncGPUReadback)
            {
                texture2DBuffer = new Texture2D(width, height, TextureFormat.RGBA32, false);
            }

            baseMat = new Mat(height, width, CvType.CV_8UC4, new Scalar(0, 0, 0, 255));
            if (baseColorFormat == OutputColorFormat)
                frameMat = baseMat.clone();
            else
                frameMat = new Mat(baseMat.rows(), baseMat.cols(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)), new Scalar(0, 0, 0, 255));

            if (Rotate90Degree)
                rotatedFrameMat = new Mat(frameMat.cols(), frameMat.rows(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)), new Scalar(0, 0, 0, 255));

            asyncGPUReadbackRequestBuffer = default;
            frameDataReadyEvent = new ManualResetEventSlim(false);

            Debug.Log("QuestPassthrough2MatHelper:: " + " width:" + frameMat.width() + " height:" + frameMat.height() + " fps:" + RequestedFPS + " useAsyncGPUReadback:" + useAsyncGPUReadback);

            yield return StartCoroutine(_GetFirstFrameCoroutine());

            frameDataReadyEvent.Set();
            didUpdateThisFrame = true;

            if (waitForEndOfFrameCoroutine != null) StopCoroutine(waitForEndOfFrameCoroutine);
            waitForEndOfFrameCoroutine = _WaitForFrameEndCoroutine();
            StartCoroutine(waitForEndOfFrameCoroutine);

            // Ensure that the cache is filled before callbacks such as _onInitialized can call GetProjectionMatrix, etc.
            UpdateCacheFromPassthrough();

            if (!autoPlayAfterInitialize)
            {
                passthroughCameraAccess.enabled = false;
                isPlaying = false;
                isPaused = false;
            }
            else
            {
                isPlaying = true;
                isPaused = false;
            }

            isInitWaiting = false;
            hasInitDone = true;
            initCoroutine = null;

            if (_onInitialized != null)
                _onInitialized.Invoke();
        }

        protected virtual IEnumerator _GetFirstFrameCoroutine()
        {
            var texture = passthroughCameraAccess?.GetTexture();
            if (texture != null)
            {
                if (useAsyncGPUReadback)
                {
                    bool isCompleted = false;
                    AsyncGPUReadbackRequest request = default;
                    AsyncGPUReadback.Request(texture, 0, (completedRequest) =>
                    {
                        request = completedRequest;
                        isCompleted = true;
                    });
                    while (!isCompleted)
                        yield return null;

                    if (!request.hasError && baseMat != null)
                    {
#if !OPENCV_DONT_USE_UNSAFE_CODE
                        OpenCVMatUtils.CopyToMat(request.GetData<byte>(), baseMat);
#endif
                        Core.flip(baseMat, baseMat, 0);
                    }
                }
                else
                {
                    OpenCVMatUtils.TextureToTexture2D(texture, texture2DBuffer);
                    yield return null;
                    if (baseMat != null)
                    {
                        OpenCVMatUtils.Texture2DToMatRaw(texture2DBuffer, baseMat);
                        Core.flip(baseMat, baseMat, 0);
                    }
                }
            }
        }

        protected virtual void CancelWaitForEndOfFrameCoroutine()
        {
            if (waitForEndOfFrameCoroutine != null)
            {
                StopCoroutine(waitForEndOfFrameCoroutine);
                ((IDisposable)waitForEndOfFrameCoroutine).Dispose();
                waitForEndOfFrameCoroutine = null;
            }
        }

        /// <summary>
        /// Return the camera to world matrix.
        /// Always returns value from cache (updated every frame alongside GetMat() when playing).
        /// </summary>
        public override Matrix4x4 GetCameraToWorldMatrix()
        {
            if (!hasInitDone || !_hasCachedValues)
                return Matrix4x4.identity;
            return Matrix4x4.TRS(_cachedPose.position, _cachedPose.rotation, Vector3.one) * Matrix4x4.Scale(new Vector3(1, 1, -1));
        }

        /// <summary>
        /// Computes sensor crop region. Mirrors PassthroughCameraAccess.CalcSensorCropRegion logic.
        /// </summary>
        private static Rect CalcSensorCropRegion(Vector2 sensorResolution, Vector2 currentResolution)
        {
            Vector2 scaleFactor = currentResolution / sensorResolution;
            scaleFactor /= Mathf.Max(scaleFactor.x, scaleFactor.y);
            return new Rect(
                sensorResolution.x * (1f - scaleFactor.x) * 0.5f,
                sensorResolution.y * (1f - scaleFactor.y) * 0.5f,
                sensorResolution.x * scaleFactor.x,
                sensorResolution.y * scaleFactor.y);
        }

        /// <summary>
        /// Updates cache with raw values from PassthroughCameraAccess. Call when passthroughCameraAccess.IsPlaying.
        /// Invoked every frame in Update() alongside CallReadback for frame coherence with GetMat().
        /// </summary>
        private void UpdateCacheFromPassthrough()
        {
            if (passthroughCameraAccess == null || !passthroughCameraAccess.IsPlaying) return;
            try
            {
                _cachedPose = passthroughCameraAccess.GetCameraPose();
                var i = passthroughCameraAccess.Intrinsics;
                _cachedFocalLength = i.FocalLength;
                _cachedPrincipalPoint = i.PrincipalPoint;
                _cachedSensorResolution = i.SensorResolution;
                _cachedCurrentResolution = passthroughCameraAccess.CurrentResolution;
                _hasCachedValues = true;
            }
            catch { /* ignore */ }
        }

        /// <summary>
        /// Computes overlay intrinsics (scaled to CurrentResolution) from cached raw values.
        /// Uses same crop/scale logic as PassthroughCameraAccess.ViewportPointToLocalRay.
        /// </summary>
        private void ComputeOverlayIntrinsicsFromCache(out Vector2 focalLength, out Vector2 principalPoint)
        {
            var sensorRes = (Vector2)_cachedSensorResolution;
            var currentRes = (Vector2)_cachedCurrentResolution;
            var crop = CalcSensorCropRegion(sensorRes, currentRes);

            float scaleX = currentRes.x / crop.width;
            float scaleY = currentRes.y / crop.height;
            focalLength = new Vector2(_cachedFocalLength.x * scaleX, _cachedFocalLength.y * scaleY);
            principalPoint = new Vector2((_cachedPrincipalPoint.x - crop.x) * scaleX, (_cachedPrincipalPoint.y - crop.y) * scaleY);
        }

        /// <summary>
        /// Return the projection matrix.
        /// Always computes from cache (updated every frame alongside GetMat() when playing).
        /// </summary>
        public override Matrix4x4 GetProjectionMatrix()
        {
            if (!hasInitDone || !_hasCachedValues)
                return Matrix4x4.identity;
            ComputeOverlayIntrinsicsFromCache(out var focalLength, out var principalPoint);
            float near = 0.01f;
            float far = 100.0f;
            return OpenCVARUtils.CalculateProjectionMatrixFromCameraMatrixValues(
                focalLength.x, focalLength.y,
                principalPoint.x, principalPoint.y,
                _cachedCurrentResolution.x, _cachedCurrentResolution.y,
                near, far);
        }

        /// <summary>
        /// Returns the camera to world transformation matrix.
        /// Always computes from cache (updated every frame alongside GetMat() when playing).
        /// </summary>
        public virtual Matrix4x4 GetCameraPoseInWorldMatrix()
        {
            if (!hasInitDone || !_hasCachedValues)
                return Matrix4x4.identity;
            return Matrix4x4.TRS(_cachedPose.position, _cachedPose.rotation, Vector3.one);
        }

        /// <summary>
        /// Gets the camera intrinsics for the specified passthrough camera eye.
        /// Returns intrinsics scaled to CurrentResolution for overlay use.
        /// Always computes from cache (updated every frame alongside GetMat() when playing).
        /// </summary>
        public virtual PassthroughCameraIntrinsics GetCameraIntrinsics()
        {
            if (!hasInitDone || !_hasCachedValues)
                return default;
            ComputeOverlayIntrinsicsFromCache(out var focalLength, out var principalPoint);
            return new PassthroughCameraIntrinsics
            {
                FocalLength = focalLength,
                PrincipalPoint = principalPoint,
                Resolution = _cachedCurrentResolution,
                Skew = 0f
            };
        }

        public override void Play()
        {
            if (hasInitDone && passthroughCameraAccess != null)
            {
                passthroughCameraAccess.enabled = true;
                isPlaying = true;
                isPaused = false;
            }
        }

        public override void Pause()
        {
            if (hasInitDone && isPlaying && passthroughCameraAccess != null)
            {
                passthroughCameraAccess.enabled = false;
                isPlaying = true;
                isPaused = true;
            }
        }

        public override void Stop()
        {
            if (hasInitDone && passthroughCameraAccess != null)
            {
                passthroughCameraAccess.enabled = false;
                isPlaying = false;
                isPaused = false;
            }
        }

        public override bool DidUpdateThisFrame()
        {
            if (!hasInitDone) return false;
            return didUpdateThisFrame;
        }

        public override Mat GetMat()
        {
            if (!hasInitDone)
            {
                return null;
            }

            if (frameMat != null && frameMat.IsDisposed)
            {
                Debug.LogWarning("QuestPassthrough2MatHelper:: " + "Please do not dispose of the Mat returned by GetMat as it will be reused");
                frameMat = new Mat(baseMat.rows(), baseMat.cols(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)));
            }
            if (rotatedFrameMat != null && rotatedFrameMat.IsDisposed)
            {
                Debug.LogWarning("QuestPassthrough2MatHelper:: " + "Please do not dispose of the Mat returned by GetMat as it will be reused");
                rotatedFrameMat = new Mat(frameMat.cols(), frameMat.rows(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)));
            }

            if (baseColorFormat == OutputColorFormat)
            {
                baseMat.copyTo(frameMat);
            }
            else
            {
                Imgproc.cvtColor(baseMat, frameMat, Source2MatHelperUtils.ColorConversionCodes(baseColorFormat, OutputColorFormat));
            }

            return Source2MatHelperUtils.ApplyMatTransformations(frameMat, rotatedFrameMat, FlipVertical, FlipHorizontal);
        }

        public override string GetDeviceName()
        {
            return hasInitDone ? "Quest_PassthroughCameraAccess" : "";
        }

        public override float GetFPS()
        {
            return hasInitDone ? (passthroughCameraAccess?.MaxFramerate ?? 60f) : -1f;
        }

        public override WebCamTexture GetWebCamTexture()
        {
            return null;
        }

        public override bool IsFrontFacing()
        {
            return false;
        }

        protected override void ReleaseResources()
        {
            CancelWaitForEndOfFrameCoroutine();

            isInitWaiting = false;
            hasInitDone = false;
            didUpdateThisFrame = false;
            isPlaying = false;
            isPaused = false;

            if (frameDataReadyEvent != null)
            {
                frameDataReadyEvent.Dispose();
                frameDataReadyEvent = null;
            }

            asyncGPUReadbackRequestBuffer = default;

            _hasCachedValues = false;

            if (passthroughCameraAccess != null)
            {
                passthroughCameraAccess.enabled = false;
                passthroughCameraAccess = null;
            }

            if (texture2DBuffer != null)
            {
                Texture2D.Destroy(texture2DBuffer);
                texture2DBuffer = null;
            }

            if (frameMat != null)
            {
                frameMat.Dispose();
                frameMat = null;
            }
            if (baseMat != null)
            {
                baseMat.Dispose();
                baseMat = null;
            }
            if (rotatedFrameMat != null)
            {
                rotatedFrameMat.Dispose();
                rotatedFrameMat = null;
            }
            if (colors != null)
                colors = null;
        }

        public override void Dispose()
        {
            if (isInitWaiting)
            {
                CancelInitCoroutine();
                CancelWaitForEndOfFrameCoroutine();
                ReleaseResources();
            }
            else if (hasInitDone)
            {
                CancelWaitForEndOfFrameCoroutine();
                ReleaseResources();
                if (_onDisposed != null)
                    _onDisposed.Invoke();
            }
        }

#endif
    }

    /// <summary>
    /// Defines the position of a passthrough camera relative to the headset
    /// </summary>
    public enum PassthroughCameraEye
    {
        Left,
        Right
    }

    /// <summary>
    /// Contains camera intrinsics, which describe physical characteristics of a passthrough camera
    /// </summary>
    public struct PassthroughCameraIntrinsics
    {
        public Vector2 FocalLength;
        public Vector2 PrincipalPoint;
        public Vector2Int Resolution;
        public float Skew;
    }
}

#endif
#endif
