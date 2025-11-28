using System;
using System.Collections.Generic;
using System.Text;

namespace QuestWithOpenCVForUnityExample
{
    public static class DebugUtils
    {
        // Private Fields
        private static Queue<long> _qRenderTick = new Queue<long>();

        private static Queue<long> _qVideoTick = new Queue<long>();

        private static Queue<long> _qTrackTick = new Queue<long>();

        private static StringBuilder _sb = new StringBuilder(1000);

        // Public Methods
        public static void RenderTick()
        {
            while (_qRenderTick.Count > 49)
            {
                _qRenderTick.Dequeue();
            }
            _qRenderTick.Enqueue(DateTime.Now.Ticks);
        }

        public static float GetRenderDeltaTime()
        {
            if (_qRenderTick.Count == 0)
            {
                return float.PositiveInfinity;
            }
            return (DateTime.Now.Ticks - _qRenderTick.Peek()) / 500000.0f;
        }

        public static void VideoTick()
        {
            while (_qVideoTick.Count > 49)
            {
                _qVideoTick.Dequeue();
            }
            _qVideoTick.Enqueue(DateTime.Now.Ticks);
        }

        public static float GetVideoDeltaTime()
        {
            if (_qVideoTick.Count == 0)
            {
                return float.PositiveInfinity;
            }
            return (DateTime.Now.Ticks - _qVideoTick.Peek()) / 500000.0f;
        }

        public static void TrackTick()
        {
            while (_qTrackTick.Count > 49)
            {
                _qTrackTick.Dequeue();
            }
            _qTrackTick.Enqueue(DateTime.Now.Ticks);
        }

        public static float GetTrackDeltaTime()
        {
            if (_qTrackTick.Count == 0)
            {
                return float.PositiveInfinity;
            }
            return (DateTime.Now.Ticks - _qTrackTick.Peek()) / 500000.0f;
        }

        public static void AddDebugStr(string str)
        {
            _sb.AppendLine(str);
        }

        public static void ClearDebugStr()
        {
            _sb.Clear();
        }

        public static string GetDebugStr()
        {
            return _sb.ToString();
        }

        public static int GetDebugStrLength()
        {
            return _sb.Length;
        }
    }
}
