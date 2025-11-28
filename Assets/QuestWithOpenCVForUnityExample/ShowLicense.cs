using System.Collections;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QuestWithOpenCVForUnityExample
{
    public class ShowLicense : MonoBehaviour
    {
        // Unity Lifecycle Methods
        private void Start()
        {

        }

        private void Update()
        {

        }

        // Public Methods
        /// <summary>
        /// Raises the back button click event.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("QuestWithOpenCVForUnityExample");
        }
    }
}
