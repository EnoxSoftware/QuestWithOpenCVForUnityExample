using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace QuestWithOpenCVForUnityExample
{
    /// <summary>
    /// QuestPassthrough2MatHelper Example
    /// An example of image processing (comic filter) using OpenCVForUnity on MetaQuest.
    /// Referring to http://dev.classmethod.jp/smartphone/opencv-manga-2/.
    /// </summary>
    [RequireComponent(typeof(QuestPassthrough2MatHelper))]
    public class QuestPassthrough2MatHelperExample : MonoBehaviour
    {
        // Public Fields
        public Toggle Rotate90DegreeToggle;
        public Toggle FlipVerticalToggle;
        public Toggle FlipHorizontalToggle;
        public bool ApplyComicFilter = false;
        public Toggle ApplyComicFilterToggle;
        public float VignetteScale = 0f;
        public Slider VignetteScaleSlider;

        [Space(10)]

        [HeaderAttribute("Debug")]

        public Text RenderFPS;
        public Text VideoFPS;
        public Text TrackFPS;
        public Text DebugStr;

        // Private Fields
        private ComicFilter _comicFilter;
        private Texture2D _texture;
        private Renderer _quadRenderer;
        private QuestPassthrough2MatHelper _webCamTextureToMatHelper;

        // Unity Lifecycle Methods
        private void Start()
        {
            _webCamTextureToMatHelper = gameObject.GetComponent<QuestPassthrough2MatHelper>();
            _webCamTextureToMatHelper.OutputColorFormat = Source2MatHelperColorFormat.RGBA;
            _webCamTextureToMatHelper.Initialize();

            // Update GUI state
            Rotate90DegreeToggle.isOn = _webCamTextureToMatHelper.Rotate90Degree;
            FlipVerticalToggle.isOn = _webCamTextureToMatHelper.FlipVertical;
            FlipHorizontalToggle.isOn = _webCamTextureToMatHelper.FlipHorizontal;
            ApplyComicFilterToggle.isOn = ApplyComicFilter;
            VignetteScaleSlider.value = VignetteScale;
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

            _quadRenderer.sharedMaterial.SetFloat("_VignetteScale", VignetteScale);

            float halfOfVerticalFov = Mathf.Atan(1.0f / projectionMatrix.m11);
            float aspectRatio = (1.0f / Mathf.Tan(halfOfVerticalFov)) / projectionMatrix.m00;
            Debug.Log("halfOfVerticalFov " + halfOfVerticalFov);
            Debug.Log("aspectRatio " + aspectRatio);

            _comicFilter = new ComicFilter(60, 120, 3);
        }

        /// <summary>
        /// Raises the source to mat helper disposed event.
        /// </summary>
        public void OnSourceToMatHelperDisposed()
        {
            Debug.Log("OnSourceToMatHelperDisposed");

            _comicFilter?.Dispose();

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
            if (_webCamTextureToMatHelper.IsPlaying() && _webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                DebugUtils.VideoTick();

                Mat rgbaMat = _webCamTextureToMatHelper.GetMat();

                if (ApplyComicFilter)
                {
                    _comicFilter.Process(rgbaMat, rgbaMat, false);
                }
                else
                {
                    Imgproc.rectangle(rgbaMat, new Point(0, 0), new Point(rgbaMat.width(), rgbaMat.height()), new Scalar(0, 0, 255, 255), 2);
                    Imgproc.putText(rgbaMat, "W:" + rgbaMat.width() + " H:" + rgbaMat.height(), new Point(5, rgbaMat.rows() - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 0, 255, 255), 2, Imgproc.LINE_AA, false);
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
        /// Raises the rotate 90 degree toggle value changed event.
        /// </summary>
        public void OnRotate90DegreeToggleValueChanged()
        {
            if (Rotate90DegreeToggle.isOn != _webCamTextureToMatHelper.Rotate90Degree)
            {
                _webCamTextureToMatHelper.Rotate90Degree = Rotate90DegreeToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the flip vertical toggle value changed event.
        /// </summary>
        public void OnFlipVerticalToggleValueChanged()
        {
            if (FlipVerticalToggle.isOn != _webCamTextureToMatHelper.FlipVertical)
            {
                _webCamTextureToMatHelper.FlipVertical = FlipVerticalToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the flip horizontal toggle value changed event.
        /// </summary>
        public void OnFlipHorizontalToggleValueChanged()
        {
            if (FlipHorizontalToggle.isOn != _webCamTextureToMatHelper.FlipHorizontal)
            {
                _webCamTextureToMatHelper.FlipHorizontal = FlipHorizontalToggle.isOn;
            }
        }

        /// <summary>
        /// Raises the apply comic filter toggle value changed event.
        /// </summary>
        public void OnApplyComicFilterToggleValueChanged()
        {
            ApplyComicFilter = ApplyComicFilterToggle.isOn;
        }

        /// <summary>
        /// Raises the vignette scale slider value changed event.
        /// </summary>
        public void OnVignetteScaleSliderValueChanged()
        {
            VignetteScale = VignetteScaleSlider.value;

            _quadRenderer?.sharedMaterial.SetFloat("_VignetteScale", VignetteScale);
        }
    }
}
