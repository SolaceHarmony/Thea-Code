import * as prod from './cjs/scheduler.production.js';
import * as dev from './cjs/scheduler.development.js';

const out = process.env.NODE_ENV === 'production' ? prod : dev;

export const {
  unstable_now,
  unstable_scheduleCallback,
  unstable_cancelCallback,
  unstable_shouldYield,
  unstable_requestPaint,
  unstable_runWithPriority,
  unstable_next,
  unstable_wrapCallback,
  unstable_getCurrentPriorityLevel,
  unstable_forceFrameRate,
  unstable_pauseExecution,
  unstable_continueExecution,
  unstable_getFirstCallbackNode,
  unstable_Profiling,
  unstable_IdlePriority,
  unstable_ImmediatePriority,
  unstable_LowPriority,
  unstable_NormalPriority,
  unstable_UserBlockingPriority,
} = out;

export default out;
