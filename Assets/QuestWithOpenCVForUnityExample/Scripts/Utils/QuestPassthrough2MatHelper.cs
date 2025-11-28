// SPDX-License-Identifier: MIT
//
// Portions of this file are derived from Meta Unity-PassthroughCameraApiSamples
// https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples (commit 9c9ffc4)
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Modifications:
// - Combined PassthroughCameraUtils.cs, PassthroughCameraPermissions.cs, PassthroughCameraDebugger.cs into one file
// - Adjusted namespaces and init flow to integrate with QuestPassthrough2MatHelper

#pragma warning disable 0067
#pragma warning disable 0618

#if !OPENCV_DONT_USE_WEBCAMTEXTURE_API
#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityIntegration.Helper.Source2Mat;
using UnityEngine;
using UnityEngine.Assertions;
using OpenCVForUnity.UnityIntegration;


#if UNITY_ANDROID
using UnityEngine.Android;
using PCD = QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat.PassthroughCameraDebugger;
#endif

namespace QuestWithOpenCVForUnity.UnityIntegration.Helper.Source2Mat
{
    /// <summary>
    /// A helper component class for obtaining camera frames from MetaQuest passthrough camera and converting them to OpenCV <c>Mat</c> format in real-time.
    /// </summary>
    /// <remarks>
    /// The <c>QuestPassthrough2MatHelper</c> class captures video frames from a device's camera using <see cref="WebCamTexture"/>
    /// and converts each frame to an OpenCV <c>Mat</c> object every frame.
    /// This component handles camera orientation, rotation, and necessary transformations to ensure the <c>Mat</c> output
    /// aligns correctly with the device's display orientation.
    ///
    /// This component is particularly useful for image processing tasks in Unity, such as computer vision applications,
    /// where real-time camera input in <c>Mat</c> format is required. It enables seamless integration of OpenCV-based
    /// image processing algorithms with Unity's camera input.
    ///
    /// <strong>Note:</strong> By setting outputColorFormat to RGBA, processing that does not include extra color conversion is performed.
    /// </remarks>
    /// <example>
    /// Attach this component to a GameObject and call <c>GetMat()</c> to retrieve the latest camera frame in <c>Mat</c> format.
    /// The helper class manages camera start/stop operations and frame updates internally.
    /// </example>
    /// <example>
    /// <code>
    /// Supported resolutions:
    /// 320 x 240
    /// 640 x 480
    /// 800 x 600
    /// 1280 x 960
    /// </code>
    /// </example>
    public class QuestPassthrough2MatHelper : WebCamTexture2MatHelper
    {
        [SerializeField] public PassthroughCameraEye Eye = PassthroughCameraEye.Left;

#if UNITY_ANDROID && !DISABLE_QUESTPASSTHROUGH_API

        /// <summary>
        /// Initialize this instance by coroutine.
        /// </summary>
        protected override IEnumerator _Initialize()
        {
            if (hasInitDone)
            {
                ReleaseResources();

                if (_onDisposed != null)
                    _onDisposed.Invoke();
            }

            isInitWaiting = true;

            // Wait one frame before starting initialization process
            yield return null;


            if (!PassthroughCameraUtils.IsSupported)
            {
                Debug.LogError($"Passthrough Camera functionality is not supported by the current device.");

                isInitWaiting = false;
                initCoroutine = null;

                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.CAMERA_DEVICE_NOT_EXIST, string.Empty);

                yield break;
            }

            // Checks camera permission state.
            IEnumerator coroutine = hasUserAuthorizedCameraPermission();
            yield return coroutine;

            if (!(bool)coroutine.Current)
            {
                Debug.LogError($"Passthrough Camera requires permission(s) {string.Join(" and ", CameraPermissions)}. Waiting for them to be granted...");

                isInitWaiting = false;
                initCoroutine = null;

                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.CAMERA_PERMISSION_DENIED, string.Empty);

                yield break;
            }

            // Check if Passhtrough is present in the scene and is enabled
            var ptLayer = FindAnyObjectByType<OVRPassthroughLayer>();
            if (ptLayer == null || !PassthroughCameraUtils.IsPassthroughEnabled())
            {
                Debug.LogError($"Passthrough must be enabled to use the Passthrough Camera API.");

                isInitWaiting = false;
                initCoroutine = null;

                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.CAMERA_PERMISSION_DENIED, string.Empty);

                yield break;
            }

            // Creates a WebCamTexture with settings closest to the requested name, resolution, and frame rate.
            var devices = WebCamTexture.devices;
            if (devices.Length == 0)
            {
                isInitWaiting = false;
                initCoroutine = null;

                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.CAMERA_DEVICE_NOT_EXIST, RequestedDeviceName);

                yield break;
            }

            if (webCamTexture == null)
            {
                if (PassthroughCameraUtils.EnsureInitialized() && PassthroughCameraUtils.CameraEyeToCameraIdMap.TryGetValue(Eye, out var cameraData))
                {
                    if (cameraData.index < devices.Length)
                    {
                        var deviceName = devices[cameraData.index].name;
                        if (RequestedHeight == 0 || RequestedWidth == 0)
                        {
                            var largestResolution = PassthroughCameraUtils.GetOutputSizes(Eye).OrderBy(static size => size.x * size.y).Last();
                            webCamTexture = new WebCamTexture(deviceName, largestResolution.x, largestResolution.y);
                        }
                        else
                        {
                            webCamTexture = new WebCamTexture(deviceName, RequestedWidth, RequestedHeight);
                        }
                    }
                }
            }

            // Starts the camera
            webCamTexture.Play();

            int initFrameCount = 0;
            bool isTimeout = false;

            while (true)
            {
                if (initFrameCount > TimeoutFrameCount)
                {
                    isTimeout = true;
                    break;
                }
                else if (webCamTexture.didUpdateThisFrame)
                {
                    Debug.Log("QuestPassthrough2MatHelper:: " + "devicename:" + webCamTexture.deviceName + " name:" + webCamTexture.name + " width:" + webCamTexture.width + " height:" + webCamTexture.height + " fps:" + webCamTexture.requestedFPS
                    + " videoRotationAngle:" + webCamTexture.videoRotationAngle + " videoVerticallyMirrored:" + webCamTexture.videoVerticallyMirrored + " isFrongFacing:" + webCamDevice.isFrontFacing);

                    if (colors == null || colors.Length != webCamTexture.width * webCamTexture.height)
                        colors = new Color32[webCamTexture.width * webCamTexture.height];

                    baseMat = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4, new Scalar(0, 0, 0, 255));

                    if (baseColorFormat == OutputColorFormat)
                    {
                        //frameMat = baseMat;
                        frameMat = baseMat.clone();
                    }
                    else
                    {
                        frameMat = new Mat(baseMat.rows(), baseMat.cols(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)), new Scalar(0, 0, 0, 255));
                    }

                    screenOrientation = Screen.orientation;
                    screenWidth = Screen.width;
                    screenHeight = Screen.height;

                    isScreenOrientationCorrect = IsScreenOrientationCorrect();
                    isVideoRotationAngleCorrect = IsVideoRotationAngleCorrect();

                    if (NeedsRotatedFrameMat(isScreenOrientationCorrect))
                        rotatedFrameMat = new Mat(frameMat.cols(), frameMat.rows(), CvType.CV_8UC(Source2MatHelperUtils.Channels(OutputColorFormat)), new Scalar(0, 0, 0, 255));

                    isInitWaiting = false;
                    hasInitDone = true;
                    initCoroutine = null;

                    if (!autoPlayAfterInitialize)
                    {
                        webCamTexture.Stop();
                        isPlaying = false;
                        isPaused = false;
                    }
                    else
                    {
                        isPlaying = true;
                        isPaused = false;
                    }

                    if (_onInitialized != null)
                        _onInitialized.Invoke();

                    break;
                }
                else
                {
                    initFrameCount++;
                    yield return null;
                }
            }

            if (isTimeout)
            {
                webCamTexture.Stop();
                webCamTexture = null;
                isInitWaiting = false;
                initCoroutine = null;

                if (_onErrorOccurred != null)
                    _onErrorOccurred.Invoke(Source2MatHelperErrorCode.TIMEOUT, string.Empty);
            }
        }

        /// <summary>
        /// Check camera permission state by coroutine.
        /// </summary>
        protected override IEnumerator hasUserAuthorizedCameraPermission()
        {
            if (HasCameraPermission != true)
            {
                AskCameraPermissions();

                // Wait until the camera permission check is completed and the result is determined
                while (HasCameraPermission == null)
                {
                    yield return null;
                }
            }
            yield return HasCameraPermission == true;
        }

        /// <summary>
        /// Return the camera to world matrix.
        /// </summary>
        /// <returns>The camera to world matrix.</returns>
        public override Matrix4x4 GetCameraToWorldMatrix()
        {
            try
            {
                if (!hasInitDone)
                {
                    return Matrix4x4.identity;
                }

                var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);
                // Convert the matrix to match the camera space (OpenGL right-handed system, forward is -Z)
                return Matrix4x4.TRS(cameraPose.position, cameraPose.rotation, Vector3.one) * Matrix4x4.Scale(new Vector3(1, 1, -1));
            }
            catch
            {
                return Matrix4x4.identity;
            }
        }

        /// <summary>
        /// Return the projection matrix matrix.
        /// The resolution value is the maximum resolution available for the camera.
        /// </summary>
        /// <returns>The projection matrix.</returns>
        public override Matrix4x4 GetProjectionMatrix()
        {
            try
            {
                if (!hasInitDone)
                {
                    return Matrix4x4.identity;
                }

                // Note: The image coordinate system of Android Camera2 API is the same as OpenCV's: origin at top-left, Y axis pointing down.
                // Get the camera intrinsics
                var intrinsics = PassthroughCameraUtils.GetCameraIntrinsics(Eye);

                // Clipping plane distances for Meta Quest 3
                float near = 0.01f;  // 1cm
                float far = 100.0f;  // 100m

                // Calculate the projection matrix in Unity's camera coordinate system
                return OpenCVARUtils.CalculateProjectionMatrixFromCameraMatrixValues(
                    intrinsics.FocalLength.x, intrinsics.FocalLength.y,
                    intrinsics.PrincipalPoint.x, intrinsics.PrincipalPoint.y,
                    intrinsics.Resolution.x, intrinsics.Resolution.y,
                    near, far);
            }
            catch
            {
                return Matrix4x4.identity;
            }
        }

        /// <summary>
        /// Returns the camera to world transformation matrix.
        /// This matrix represents the passthrough camera's pose (position and rotation) in world space,
        /// equivalent to Unity's Camera.transform.localToWorldMatrix.
        /// </summary>
        /// <returns>The camera to world transformation matrix.</returns>
        public virtual Matrix4x4 GetCameraPoseInWorldMatrix()
        {
            try
            {
                if (!hasInitDone)
                {
                    return Matrix4x4.identity;
                }

                var cameraPose = PassthroughCameraUtils.GetCameraPoseInWorld(Eye);
                return Matrix4x4.TRS(cameraPose.position, cameraPose.rotation, Vector3.one);
            }
            catch
            {
                return Matrix4x4.identity;
            }
        }

        /// <summary>
        /// Gets the camera intrinsics for the specified passthrough camera eye.
        /// Returns camera intrinsics including focal length, principal point, resolution, and skew coefficient.
        /// </summary>
        /// <returns>The camera intrinsics, or default values if initialization is not complete or an error occurs.</returns>
        public virtual PassthroughCameraIntrinsics GetCameraIntrinsics()
        {
            try
            {
                if (!hasInitDone)
                {
                    return default(PassthroughCameraIntrinsics);
                }

                return PassthroughCameraUtils.GetCameraIntrinsics(Eye);
            }
            catch
            {
                return default(PassthroughCameraIntrinsics);
            }
        }

#endif

        // ===== Begin: PassthroughCameraPermissions.cs (MIT, Meta) =====
        // Source: https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples/blob/9c9ffc4409a35b2d262c062c871cd676fbbc7817/Assets/PassthroughCameraApiSamples/PassthroughCamera/Scripts/PassthroughCameraPermissions.cs
        // Modifications: minor refactors for namespace and access modifiers
        [SerializeField] public List<string> PermissionRequestsOnStartup = new() { OVRPermissionsRequester.ScenePermission };

        public static readonly string[] CameraPermissions =
        {
            "android.permission.CAMERA",          // Required to use WebCamTexture object.
            "horizonos.permission.HEADSET_CAMERA" // Required to access the Passthrough Camera API in Horizon OS v74 and above.
        };

        public static bool? HasCameraPermission { get; private set; }
        private static bool s_askedOnce;

#if UNITY_ANDROID
        /// <summary>
        /// Request camera permission if the permission is not authorized by the user.
        /// </summary>
        public void AskCameraPermissions()
        {
            if (s_askedOnce)
            {
                return;
            }
            s_askedOnce = true;
            if (IsAllCameraPermissionsGranted())
            {
                HasCameraPermission = true;
                PCD.DebugMessage(LogType.Log, "PCA: All camera permissions granted.");
            }
            else
            {
                PCD.DebugMessage(LogType.Log, "PCA: Requesting camera permissions.");

                var callbacks = new PermissionCallbacks();
                callbacks.PermissionDenied += PermissionCallbacksPermissionDenied;
                callbacks.PermissionGranted += PermissionCallbacksPermissionGranted;
                callbacks.PermissionDeniedAndDontAskAgain += PermissionCallbacksPermissionDenied;

                // It's important to request all necessary permissions in one request because only one 'PermissionCallbacks' instance is supported at a time.
                var allPermissions = CameraPermissions.Concat(PermissionRequestsOnStartup).ToArray();
                Permission.RequestUserPermissions(allPermissions, callbacks);
            }
        }

        /// <summary>
        /// Permission Granted callback
        /// </summary>
        /// <param name="permissionName"></param>
        private static void PermissionCallbacksPermissionGranted(string permissionName)
        {
            PCD.DebugMessage(LogType.Log, $"PCA: Permission {permissionName} Granted");

            // Only initialize the WebCamTexture object if both permissions are granted
            if (IsAllCameraPermissionsGranted())
            {
                HasCameraPermission = true;
            }
        }

        /// <summary>
        /// Permission Denied callback.
        /// </summary>
        /// <param name="permissionName"></param>
        private static void PermissionCallbacksPermissionDenied(string permissionName)
        {
            PCD.DebugMessage(LogType.Warning, $"PCA: Permission {permissionName} Denied");
            HasCameraPermission = false;
            s_askedOnce = false;
        }

        private static bool IsAllCameraPermissionsGranted() => CameraPermissions.All(Permission.HasUserAuthorizedPermission);
#endif
        // ===== End: PassthroughCameraPermissions.cs =====
    }

    /// <summary>
    /// Defines the position of a passthrough camera relative to the headset
    /// </summary>
    public enum PassthroughCameraEye
    {
        Left,
        Right
    }

    // ===== Begin: PassthroughCameraDebugger.cs (MIT, Meta) =====
    // Source: https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples/blob/9c9ffc4409a35b2d262c062c871cd676fbbc7817/Assets/PassthroughCameraApiSamples/PassthroughCamera/Scripts/PassthroughCameraDebugger.cs
    // Modifications: minor refactors for namespace and access modifiers
    public static class PassthroughCameraDebugger
    {
        public enum DebuglevelEnum
        {
            ALL,
            NONE,
            ONLY_ERROR,
            ONLY_LOG,
            ONLY_WARNING
        }

        public static DebuglevelEnum DebugLevel = DebuglevelEnum.ALL;

        /// <summary>
        /// Send debug information to Unity console based on DebugType and DebugLevel
        /// </summary>
        /// <param name="mType"></param>
        /// <param name="message"></param>
        public static void DebugMessage(LogType mType, string message)
        {
            switch (mType)
            {
                case LogType.Error:
                    if (DebugLevel is DebuglevelEnum.ALL or DebuglevelEnum.ONLY_ERROR)
                    {
                        Debug.LogError(message);
                    }
                    break;
                case LogType.Log:
                    if (DebugLevel is DebuglevelEnum.ALL or DebuglevelEnum.ONLY_LOG)
                    {
                        Debug.Log(message);
                    }
                    break;
                case LogType.Warning:
                    if (DebugLevel is DebuglevelEnum.ALL or DebuglevelEnum.ONLY_WARNING)
                    {
                        Debug.LogWarning(message);
                    }
                    break;
            }
        }
    }
    // ===== End: PassthroughCameraDebugger.cs =====

    // ===== Begin: PassthroughCameraUtils.cs (MIT, Meta) =====
    // Source: https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples/blob/9c9ffc4409a35b2d262c062c871cd676fbbc7817/Assets/PassthroughCameraApiSamples/PassthroughCamera/Scripts/PassthroughCameraUtils.cs
    // Modifications: minor refactors for namespace and access modifiers
    public static class PassthroughCameraUtils
    {
        // The Horizon OS starts supporting PCA with v74.
        public const int MINSUPPORTOSVERSION = 74;

        // The only pixel format supported atm
        private const int YUV_420_888 = 0x00000023;

        private static AndroidJavaObject s_currentActivity;
        private static AndroidJavaObject s_cameraManager;
        private static bool? s_isSupported;
        private static int? s_horizonOsVersion;

        // Caches
        internal static readonly Dictionary<PassthroughCameraEye, (string id, int index)> CameraEyeToCameraIdMap = new();
        private static readonly ConcurrentDictionary<PassthroughCameraEye, List<Vector2Int>> s_cameraOutputSizes = new();
        private static readonly ConcurrentDictionary<string, AndroidJavaObject> s_cameraCharacteristicsMap = new();
        private static readonly OVRPose?[] s_cachedCameraPosesRelativeToHead = new OVRPose?[2];

        /// <summary>
        /// Get the Horizon OS version number on the headset
        /// </summary>
        public static int? HorizonOSVersion
        {
            get
            {
                if (!s_horizonOsVersion.HasValue)
                {
                    var vrosClass = new AndroidJavaClass("vros.os.VrosBuild");
                    s_horizonOsVersion = vrosClass.CallStatic<int>("getSdkVersion");
#if OVR_INTERNAL_CODE
                    // 10000 means that the build doesn't have a proper release version, and it is still in Mainline,
                    // not in a release branch.
#endif // OVR_INTERNAL_CODE
                    if (s_horizonOsVersion == 10000)
                    {
                        s_horizonOsVersion = -1;
                    }
                }

                return s_horizonOsVersion.Value != -1 ? s_horizonOsVersion.Value : null;
            }
        }

        /// <summary>
        /// Returns true if the current headset supports Passthrough Camera API
        /// </summary>
        public static bool IsSupported
        {
            get
            {
                if (!s_isSupported.HasValue)
                {
                    var headset = OVRPlugin.GetSystemHeadsetType();
                    return (headset == OVRPlugin.SystemHeadset.Meta_Quest_3 ||
                            headset == OVRPlugin.SystemHeadset.Meta_Quest_3S) &&
                           (!HorizonOSVersion.HasValue || HorizonOSVersion >= MINSUPPORTOSVERSION);
                }

                return s_isSupported.Value;
            }
        }

        /// <summary>
        /// Provides a list of resolutions supported by the passthrough camera. Developers should use one of those
        /// when initializing the camera.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        public static List<Vector2Int> GetOutputSizes(PassthroughCameraEye cameraEye)
        {
            return s_cameraOutputSizes.GetOrAdd(cameraEye, GetOutputSizesInternal(cameraEye));
        }

        /// <summary>
        /// Returns the camera intrinsics for a specified passthrough camera. All the intrinsics values are provided
        /// in pixels. The resolution value is the maximum resolution available for the camera.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        public static PassthroughCameraIntrinsics GetCameraIntrinsics(PassthroughCameraEye cameraEye)
        {
            var cameraCharacteristics = GetCameraCharacteristics(cameraEye);
            var intrinsicsArr = GetCameraValueByKey<float[]>(cameraCharacteristics, "LENS_INTRINSIC_CALIBRATION");

            // Querying the camera resolution for which the intrinsics are provided
            // https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE
            // This is a Rect of 4 elements: [bottom, left, right, top] with (0,0) at top-left corner.
            using var sensorSize = GetCameraValueByKey<AndroidJavaObject>(cameraCharacteristics, "SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE");

            return new PassthroughCameraIntrinsics
            {
                FocalLength = new Vector2(intrinsicsArr[0], intrinsicsArr[1]),
                PrincipalPoint = new Vector2(intrinsicsArr[2], intrinsicsArr[3]),
                Resolution = new Vector2Int(sensorSize.Get<int>("right"), sensorSize.Get<int>("bottom")),
                Skew = intrinsicsArr[4]
            };
        }

        /// <summary>
        /// Returns an Android Camera2 API's cameraId associated with the passthrough camera specified in the argument.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        /// <exception cref="ApplicationException">Throws an exception if the code was not able to find cameraId</exception>
        public static string GetCameraIdByEye(PassthroughCameraEye cameraEye)
        {
            _ = EnsureInitialized();

            return !CameraEyeToCameraIdMap.TryGetValue(cameraEye, out var value)
                ? throw new ApplicationException($"Cannot find cameraId for the eye {cameraEye}")
                : value.id;
        }

        /// <summary>
        /// Returns the world pose of a passthrough camera at a given time.
        /// The LENS_POSE_TRANSLATION and LENS_POSE_ROTATION keys in 'android.hardware.camera2' are relative to the origin, so they can be cached to improve performance.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        /// <returns>The passthrough camera's world pose</returns>
        public static Pose GetCameraPoseInWorld(PassthroughCameraEye cameraEye)
        {
            var index = cameraEye == PassthroughCameraEye.Left ? 0 : 1;

            if (s_cachedCameraPosesRelativeToHead[index] == null)
            {
                var cameraId = GetCameraIdByEye(cameraEye);
                using var cameraCharacteristics = s_cameraManager.Call<AndroidJavaObject>("getCameraCharacteristics", cameraId);

                var cameraTranslation = GetCameraValueByKey<float[]>(cameraCharacteristics, "LENS_POSE_TRANSLATION");
                var p_headFromCamera = new Vector3(cameraTranslation[0], cameraTranslation[1], -cameraTranslation[2]);

                var cameraRotation = GetCameraValueByKey<float[]>(cameraCharacteristics, "LENS_POSE_ROTATION");
                var q_cameraFromHead = new Quaternion(-cameraRotation[0], -cameraRotation[1], cameraRotation[2], cameraRotation[3]);

                var q_headFromCamera = Quaternion.Inverse(q_cameraFromHead);

                s_cachedCameraPosesRelativeToHead[index] = new OVRPose
                {
                    position = p_headFromCamera,
                    orientation = q_headFromCamera
                };
            }

            var headFromCamera = s_cachedCameraPosesRelativeToHead[index].Value;
            var worldFromHead = OVRPlugin.GetNodePoseStateImmediate(OVRPlugin.Node.Head).Pose.ToOVRPose();
            var worldFromCamera = worldFromHead * headFromCamera;
            worldFromCamera.orientation *= Quaternion.Euler(180, 0, 0);

            return new Pose(worldFromCamera.position, worldFromCamera.orientation);
        }

        /// <summary>
        /// Returns a 3D ray in the world space which starts from the passthrough camera origin and passes through the
        /// 2D camera pixel.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        /// <param name="screenPoint">A 2D point on the camera texture. The point is positioned relative to the
        ///     maximum available camera resolution. This resolution can be obtained using <see cref="GetCameraIntrinsics"/>
        ///     or <see cref="GetOutputSizes"/> methods.
        /// </param>
        public static Ray ScreenPointToRayInWorld(PassthroughCameraEye cameraEye, Vector2Int screenPoint)
        {
            var rayInCamera = ScreenPointToRayInCamera(cameraEye, screenPoint);
            var cameraPoseInWorld = GetCameraPoseInWorld(cameraEye);
            var rayDirectionInWorld = cameraPoseInWorld.rotation * rayInCamera.direction;
            return new Ray(cameraPoseInWorld.position, rayDirectionInWorld);
        }

        /// <summary>
        /// Returns a 3D ray in the camera space which starts from the passthrough camera origin - which is always
        /// (0, 0, 0) - and passes through the 2D camera pixel.
        /// </summary>
        /// <param name="cameraEye">The passthrough camera</param>
        /// <param name="screenPoint">A 2D point on the camera texture. The point is positioned relative to the
        /// maximum available camera resolution. This resolution can be obtained using <see cref="GetCameraIntrinsics"/>
        /// or <see cref="GetOutputSizes"/> methods.
        /// </param>
        public static Ray ScreenPointToRayInCamera(PassthroughCameraEye cameraEye, Vector2Int screenPoint)
        {
            var intrinsics = GetCameraIntrinsics(cameraEye);
            var directionInCamera = new Vector3
            {
                x = (screenPoint.x - intrinsics.PrincipalPoint.x) / intrinsics.FocalLength.x,
                y = (screenPoint.y - intrinsics.PrincipalPoint.y) / intrinsics.FocalLength.y,
                z = 1
            };

            return new Ray(Vector3.zero, directionInCamera);
        }

        #region Private methods

        internal static bool EnsureInitialized()
        {
            if (CameraEyeToCameraIdMap.Count == 2)
            {
                return true;
            }

            Debug.Log($"PCA: PassthroughCamera - Initializing...");
            using var activityClass = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
            s_currentActivity = activityClass.GetStatic<AndroidJavaObject>("currentActivity");
            s_cameraManager = s_currentActivity.Call<AndroidJavaObject>("getSystemService", "camera");
            Assert.IsNotNull(s_cameraManager, "Camera manager has not been provided by the Android system");

            var cameraIds = GetCameraIdList();
            Debug.Log($"PCA: PassthroughCamera - cameraId list is {string.Join(", ", cameraIds)}");

            for (var idIndex = 0; idIndex < cameraIds.Length; idIndex++)
            {
                var cameraId = cameraIds[idIndex];
                CameraSource? cameraSource = null;
                CameraPosition? cameraPosition = null;

                var cameraCharacteristics = GetCameraCharacteristics(cameraId);
                using var keysList = cameraCharacteristics.Call<AndroidJavaObject>("getKeys");
                var size = keysList.Call<int>("size");
                for (var i = 0; i < size; i++)
                {
                    using var key = keysList.Call<AndroidJavaObject>("get", i);
                    var keyName = key.Call<string>("getName");

                    if (string.Equals(keyName, "com.meta.extra_metadata.camera_source", StringComparison.OrdinalIgnoreCase))
                    {
                        // Both `com.meta.extra_metadata.camera_source` and `com.meta.extra_metadata.camera_source` are
                        // custom camera fields which are stored as arrays of size 1, instead of single values.
                        // We have to read those values correspondingly
                        var cameraSourceArr = GetCameraValueByKey<sbyte[]>(cameraCharacteristics, key);
                        if (cameraSourceArr == null || cameraSourceArr.Length != 1)
                            continue;

                        cameraSource = (CameraSource)cameraSourceArr[0];
                    }
                    else if (string.Equals(keyName, "com.meta.extra_metadata.position", StringComparison.OrdinalIgnoreCase))
                    {
                        var cameraPositionArr = GetCameraValueByKey<sbyte[]>(cameraCharacteristics, key);
                        if (cameraPositionArr == null || cameraPositionArr.Length != 1)
                            continue;

                        cameraPosition = (CameraPosition)cameraPositionArr[0];
                    }
                }

                if (!cameraSource.HasValue || !cameraPosition.HasValue || cameraSource.Value != CameraSource.Passthrough)
                    continue;

                switch (cameraPosition)
                {
                    case CameraPosition.Left:
                        Debug.Log($"PCA: Found left passthrough cameraId = {cameraId}");
                        CameraEyeToCameraIdMap[PassthroughCameraEye.Left] = (cameraId, idIndex);
                        break;
                    case CameraPosition.Right:
                        Debug.Log($"PCA: Found right passthrough cameraId = {cameraId}");
                        CameraEyeToCameraIdMap[PassthroughCameraEye.Right] = (cameraId, idIndex);
                        break;
                    default:
                        throw new ApplicationException($"Cannot parse Camera Position value {cameraPosition}");
                }
            }

            return CameraEyeToCameraIdMap.Count == 2;
        }

        internal static bool IsPassthroughEnabled()
        {
            return OVRManager.IsInsightPassthroughSupported() &&
                OVRManager.IsInsightPassthroughInitialized() &&
                OVRManager.instance.isInsightPassthroughEnabled;
        }

        private static string[] GetCameraIdList()
        {
            return s_cameraManager.Call<string[]>("getCameraIdList");
        }

        private static List<Vector2Int> GetOutputSizesInternal(PassthroughCameraEye cameraEye)
        {
            _ = EnsureInitialized();

            var cameraId = GetCameraIdByEye(cameraEye);
            var cameraCharacteristics = GetCameraCharacteristics(cameraId);
            using var configurationMap =
                GetCameraValueByKey<AndroidJavaObject>(cameraCharacteristics, "SCALER_STREAM_CONFIGURATION_MAP");
            var outputSizes = configurationMap.Call<AndroidJavaObject[]>("getOutputSizes", YUV_420_888);

            var result = new List<Vector2Int>();
            foreach (var outputSize in outputSizes)
            {
                var width = outputSize.Call<int>("getWidth");
                var height = outputSize.Call<int>("getHeight");
                result.Add(new Vector2Int(width, height));
            }

            foreach (var obj in outputSizes)
            {
                obj?.Dispose();
            }

            return result;
        }

        private static AndroidJavaObject GetCameraCharacteristics(string cameraId)
        {
            return s_cameraCharacteristicsMap.GetOrAdd(cameraId,
                _ => s_cameraManager.Call<AndroidJavaObject>("getCameraCharacteristics", cameraId));
        }

        private static AndroidJavaObject GetCameraCharacteristics(PassthroughCameraEye eye)
        {
            var cameraId = GetCameraIdByEye(eye);
            return GetCameraCharacteristics(cameraId);
        }

        private static T GetCameraValueByKey<T>(AndroidJavaObject cameraCharacteristics, string keyStr)
        {
            using var key = cameraCharacteristics.GetStatic<AndroidJavaObject>(keyStr);
            return GetCameraValueByKey<T>(cameraCharacteristics, key);
        }

        private static T GetCameraValueByKey<T>(AndroidJavaObject cameraCharacteristics, AndroidJavaObject key)
        {
            return cameraCharacteristics.Call<T>("get", key);
        }

        private enum CameraSource
        {
            Passthrough = 0
        }

        private enum CameraPosition
        {
            Left = 0,
            Right = 1
        }

        #endregion Private methods
    }

    /// <summary>
    /// Contains camera intrinsics, which describe physical characteristics of a passthrough camera
    /// </summary>
    public struct PassthroughCameraIntrinsics
    {
        /// <summary>
        /// The focal length in pixels
        /// </summary>
        public Vector2 FocalLength;
        /// <summary>
        /// The principal point from the top-left corner of the image, expressed in pixels
        /// </summary>
        public Vector2 PrincipalPoint;
        /// <summary>
        /// The resolution in pixels for which the intrinsics are defined
        /// </summary>
        public Vector2Int Resolution;
        /// <summary>
        /// The skew coefficient which represents the non-perpendicularity of the image sensor's x and y axes
        /// </summary>
        public float Skew;
    }
    // ===== End: PassthroughCameraUtils.cs =====
}

#endif
#endif
