using System.Collections;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityIntegration;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace QuestWithOpenCVForUnityExample
{
    /// <summary>
    /// QuestWithOpenCVForUnity Example
    /// </summary>
    public class QuestWithOpenCVForUnityExample : MonoBehaviour
    {
        // Public Fields
        public Text ExampleTitle;
        public Text VersionInfo;
        public ScrollRect ScrollRect;

        // Private Fields
        private static float _verticalNormalizedPosition = 1f;

        // Unity Lifecycle Methods
        private void Start()
        {
            ExampleTitle.text = "QuestWithOpenCVForUnity Example " + Application.version;

            VersionInfo.text = Core.NATIVE_LIBRARY_NAME + " " + OpenCVEnv.GetVersion() + " (" + Core.VERSION + ")";
            VersionInfo.text += " / UnityEditor " + Application.unityVersion;
            VersionInfo.text += " / ";

#if UNITY_EDITOR
            VersionInfo.text += "Editor";
#elif UNITY_STANDALONE_WIN
            VersionInfo.text += "Windows";
#elif UNITY_STANDALONE_OSX
            VersionInfo.text += "Mac OSX";
#elif UNITY_STANDALONE_LINUX
            VersionInfo.text += "Linux";
#elif UNITY_ANDROID
            VersionInfo.text += "Android";
#elif UNITY_IOS
            VersionInfo.text += "iOS";
#elif UNITY_WSA
            VersionInfo.text += "WSA";
#elif UNITY_WEBGL
            VersionInfo.text += "WebGL";
#endif
            VersionInfo.text += " ";
#if ENABLE_MONO
            VersionInfo.text += "Mono";
#elif ENABLE_IL2CPP
            VersionInfo.text += "IL2CPP";
#elif ENABLE_DOTNET
            VersionInfo.text += ".NET";
#endif

            ScrollRect.verticalNormalizedPosition = _verticalNormalizedPosition;
        }

        private void Update()
        {

        }

        // Public Methods
        public void OnScrollRectValueChanged()
        {
            _verticalNormalizedPosition = ScrollRect.verticalNormalizedPosition;
        }


        public void OnShowLicenseButtonClick()
        {
            SceneManager.LoadScene("ShowLicense");
        }

        public void OnQuestPassthrough2MatHelperExampleButtonClick()
        {
            SceneManager.LoadScene("QuestPassthrough2MatHelperExample");
        }

        public void OnQuestArUcoExampleButtonClick()
        {
            SceneManager.LoadScene("QuestArUcoExample");
        }

        public void OnQuestMultiObjectTrackingExampleButtonClick()
        {
            SceneManager.LoadScene("QuestMultiObjectTrackingExample");
        }

        public void OnQuestFaceIdentificationEstimatorExampleButtonClick()
        {
            SceneManager.LoadScene("QuestFaceIdentificationEstimatorExample");
        }
    }
}
