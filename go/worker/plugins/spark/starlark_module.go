package spark

import (
	"fmt"
	"time"

	"github.com/cadence-workflow/starlark-worker/ext"
	"github.com/cadence-workflow/starlark-worker/service"
	"github.com/cadence-workflow/starlark-worker/star"
	"github.com/cadence-workflow/starlark-worker/workflow"
	"github.com/michelangelo-ai/michelangelo/go/worker/activities/spark"
	"github.com/michelangelo-ai/michelangelo/go/worker/plugins/utils"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
	"go.starlark.net/starlark"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// These are some error reasons
const (
	errorReasonUnpackArgs           = "UnpackArgsError"
	errorReasonConvertJob           = "ConvertSparkJobError"
	errorReasonConvertStarlarkValue = "ConvertStarlarkValueError"
	errorReasonSubmitJob            = "SubmitJobError"
	errorReasonSensorJob            = "SensorJobError"
	errorReasonTermninateJob        = "TerminateJobError"
)

const reasonForCancel = "Canceled by request"

// These are general const
const (
	defaultPollSeconds  = 10
	maxJobSensorRetries = 100
)

// TODO(#561): andrii: implement Spark starlark plugin here

var _ starlark.HasAttrs = (*module)(nil)
var poll int64 = 10

type module struct {
	attributes map[string]*starlark.Builtin
	properties map[string]star.PropertyFactory
}

func newModule() starlark.Value {
	m := &module{}
	m.attributes = map[string]*starlark.Builtin{
		"create_job": starlark.NewBuiltin("create_job", m.createJob),
		"sensor_job": starlark.NewBuiltin("sensor_job", m.sensorJob),
		"run_job":    starlark.NewBuiltin("run_job", m.runJob),
	}
	m.properties = map[string]star.PropertyFactory{
		"running_condition_type":   getRunningConditionType,
		"succeeded_condition_type": getSucceededConditionType,
		"killed_condition_type":    getKilledConditionType,
	}
	return m
}

func (r *module) String() string        { return pluginID }
func (r *module) Type() string          { return pluginID }
func (r *module) Freeze()               {}
func (r *module) Truth() starlark.Bool  { return true }
func (r *module) Hash() (uint32, error) { return 0, fmt.Errorf("no-hash") }
func (r *module) Attr(n string) (starlark.Value, error) {
	return star.Attr(
		r, n, r.attributes, r.properties)
}
func (r *module) AttrNames() []string { return ext.SortedKeys(r.attributes) }

func (r *module) doCreateJob(ctx workflow.Context, sparkJob *v2pb.SparkJob, timeout int64) (*spark.CreateSparkJobActivityResponse, error) {
	logger := workflow.GetLogger(ctx)

	srp := utils.DefaultRetryPolicy
	srp.ExpirationInterval = time.Second * time.Duration(timeout)
	srp.InitialInterval = time.Second * time.Duration(poll)
	createCtx := workflow.WithRetryPolicy(ctx, srp)

	var createRes spark.CreateSparkJobActivityResponse
	if err := workflow.ExecuteActivity(createCtx, spark.Activities.CreateSparkJob, v2pb.CreateSparkJobRequest{
		SparkJob: sparkJob,
	}).Get(ctx, &createRes); err != nil {
		logger.Error("builtin-error", ext.ZapError(err)...)
		return nil, err
	}
	return &createRes, nil
}

func (r *module) doSensorJob(ctx workflow.Context, sparkJob *v2pb.SparkJob, timeout int64, pollInterval int, assertConditionType string) (*spark.SensorSparkJobResponse, error) {
	logger := workflow.GetLogger(ctx)

	srp := utils.DefaultSensorRetryPolicy
	srp.ExpirationInterval = time.Second * time.Duration(timeout)
	srp.InitialInterval = time.Second * time.Duration(pollInterval)
	sensorCtx := workflow.WithRetryPolicy(ctx, srp)

	getSparkJobRequest := v2pb.GetSparkJobRequest{
		Name:      sparkJob.Name,
		Namespace: sparkJob.Namespace,
	}
	var getSparkJobResponse spark.SensorSparkJobResponse
	maxSensorTries := maxJobSensorRetries
	for i := 0; i < maxSensorTries; i++ {
		if err := workflow.ExecuteActivity(sensorCtx, spark.Activities.SensorSparkJob, getSparkJobRequest).Get(ctx, &getSparkJobResponse); err != nil {
			if workflow.IsCanceledError(ctx, err) {
				ctx, _ = workflow.NewDisconnectedContext(ctx)
				terminateRequest := spark.TerminateSparkJobRequest{
					Name:      sparkJob.Name,
					Namespace: sparkJob.Namespace,
					Type:      v2pb.TERMINATION_TYPE_FAILED,
					Reason:    reasonForCancel,
				}
				var terminateResponse v2pb.UpdateSparkJobResponse
				if terminateErr := workflow.ExecuteActivity(ctx, spark.Activities.TerminateSparkJob, terminateRequest).Get(ctx, &terminateResponse); terminateErr != nil {
					logger.Error(errorReasonTermninateJob, ext.ZapError(terminateErr)...)
					return nil, terminateErr
				}
				return &spark.SensorSparkJobResponse{
					SparkJob: terminateResponse.SparkJob,
					Terminal: true,
				}, nil
			}
			logger.Error(errorReasonSensorJob, ext.ZapError(err)...)
			continue
		}
		if getSparkJobResponse.Terminal {
			break
		}
	}
	return &getSparkJobResponse, nil
}

func (r *module) createJob(t *starlark.Thread, _ *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	ctx := service.GetContext(t)
	logger := workflow.GetLogger(ctx)

	var _job *starlark.Dict
	var timeout int64

	if err := starlark.UnpackArgs("create_job", args, kwargs,
		"job", &_job,
		"timeout_seconds?", &timeout,
	); err != nil {
		logger.Error(errorReasonUnpackArgs, ext.ZapError(err)...)
		return nil, err
	}
	if timeout == 0 {
		timeout = int64(utils.LongTimeout.Seconds())
	}

	var sparkJob v2pb.SparkJob
	if err := utils.AsGo(_job, &sparkJob); err != nil {
		logger.Error("builtin-error", ext.ZapError(err)...)
		return nil, err
	}

	createRes, err := r.doCreateJob(ctx, &sparkJob, timeout)
	if err != nil {
		return nil, err
	}

	enhancedResponse := map[string]interface{}{
		"sparkJob":   createRes.SparkJob,
		"activityId": createRes.ActivityID,
	}

	var res starlark.Value
	if err := utils.AsStar(enhancedResponse, &res); err != nil {
		logger.Error("builtin-error", ext.ZapError(err)...)
		return nil, err
	}

	return res, nil
}

// waits till a specific condition is meet (blocking call) .
//
//	sensor_job(job, timeout_seconds=0, poll_seconds=10, assert_condition_type="succeeded") -> job
//
//	  job: a spark job crd in json format
//	  timeout_seconds: int: job is expected to finish within the given time
//	  poll_seconds: int: job status poll interval
//
//	  return: dict: job status
func (r *module) sensorJob(t *starlark.Thread, _ *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	ctx := service.GetContext(t)
	logger := workflow.GetLogger(ctx)

	var _job *starlark.Dict
	timeout := int64(utils.LongTimeout.Seconds())
	poll := defaultPollSeconds
	var assertConditionType string = utils.SucceededCondition

	if err := starlark.UnpackArgs("sensor_job", args, kwargs,
		"job", &_job,
		"assert_condition_type?", &assertConditionType,
	); err != nil {
		logger.Error(errorReasonUnpackArgs, ext.ZapError(err)...)
		return nil, err
	}
	var sparkJob v2pb.SparkJob
	if err := utils.AsGo(_job, &sparkJob); err != nil {
		logger.Error(errorReasonConvertJob, ext.ZapError(err)...)
		return nil, err
	}

	sensorRes, err := r.doSensorJob(ctx, &sparkJob, timeout, poll, assertConditionType)
	if err != nil {
		return nil, err
	}

	var sparkJobValue starlark.Value
	if err := utils.AsStar(sensorRes.SparkJob, &sparkJobValue); err != nil {
		logger.Error(errorReasonConvertStarlarkValue, ext.ZapError(err)...)
		return nil, err
	}
	return sparkJobValue, nil
}

// run_job creates a spark job from flat parameters and waits for it to reach a terminal state.
// The kwargs match run_spark_job()'s Python signature exactly, so the transpiled
// __spark__.run_job(...) call works without restructuring.
//
//	run_job(namespace, main_application_file, main_class=None, args=None, image=None,
//	        driver_cpu=None, driver_memory=None, executor_cpu=None, executor_memory=None,
//	        executor_instances=None, spark_conf=None, deps_jars=None, deps_py_files=None,
//	        spark_version="3.5.5", timeout_seconds=0, poll_seconds=10) -> job
//
//	  return: dict: final job status
func (r *module) runJob(t *starlark.Thread, _ *starlark.Builtin, args starlark.Tuple, kwargs []starlark.Tuple) (starlark.Value, error) {
	ctx := service.GetContext(t)
	logger := workflow.GetLogger(ctx)

	var namespace string
	var mainApplicationFile string
	var mainClass string
	var mainArgs *starlark.List
	var image string
	var driverCPU int
	var driverMemory string
	var executorCPU int
	var executorMemory string
	var executorInstances int
	var sparkConf *starlark.Dict
	var depsJars *starlark.List
	var depsPyFiles *starlark.List
	var sparkVersion string = "3.5.5"
	var timeout int64
	pollSeconds := defaultPollSeconds
	var retryAttempts int

	if err := starlark.UnpackArgs("run_job", args, kwargs,
		"namespace", &namespace,
		"main_application_file", &mainApplicationFile,
		"main_class?", &mainClass,
		"args?", &mainArgs,
		"image?", &image,
		"driver_cpu?", &driverCPU,
		"driver_memory?", &driverMemory,
		"executor_cpu?", &executorCPU,
		"executor_memory?", &executorMemory,
		"executor_instances?", &executorInstances,
		"spark_conf?", &sparkConf,
		"deps_jars?", &depsJars,
		"deps_py_files?", &depsPyFiles,
		"spark_version?", &sparkVersion,
		"timeout_seconds?", &timeout,
		"poll_seconds?", &pollSeconds,
		"retry_attempts?", &retryAttempts,
	); err != nil {
		logger.Error(errorReasonUnpackArgs, ext.ZapError(err)...)
		return nil, err
	}
	if timeout == 0 {
		timeout = int64(utils.LongTimeout.Seconds())
	}

	sparkJob := &v2pb.SparkJob{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    namespace,
			GenerateName: "uniflow-splg-",
		},
		Spec: v2pb.SparkJobSpec{
			MainApplicationFile: mainApplicationFile,
			MainClass:           mainClass,
			SparkVersion:        sparkVersion,
			Driver: &v2pb.DriverSpec{
				Pod: &v2pb.PodSpec{
					Resource: &v2pb.ResourceSpec{
						Cpu:    int32(driverCPU),
						Memory: driverMemory,
					},
					Image: image,
				},
			},
			Executor: &v2pb.ExecutorSpec{
				Pod: &v2pb.PodSpec{
					Resource: &v2pb.ResourceSpec{
						Cpu:    int32(executorCPU),
						Memory: executorMemory,
					},
					Image: image,
				},
				Instances: int32(executorInstances),
			},
		},
	}

	if mainArgs != nil {
		for i := 0; i < mainArgs.Len(); i++ {
			if s, ok := mainArgs.Index(i).(starlark.String); ok {
				sparkJob.Spec.MainArgs = append(sparkJob.Spec.MainArgs, string(s))
			}
		}
	}

	if sparkConf != nil {
		sparkJob.Spec.SparkConf = make(map[string]string)
		for _, item := range sparkConf.Items() {
			if k, ok := item[0].(starlark.String); ok {
				if v, ok := item[1].(starlark.String); ok {
					sparkJob.Spec.SparkConf[string(k)] = string(v)
				}
			}
		}
	}

	deps := &v2pb.Dependencies{}
	if depsJars != nil {
		for i := 0; i < depsJars.Len(); i++ {
			if s, ok := depsJars.Index(i).(starlark.String); ok {
				deps.Jars = append(deps.Jars, string(s))
			}
		}
	}
	if depsPyFiles != nil {
		for i := 0; i < depsPyFiles.Len(); i++ {
			if s, ok := depsPyFiles.Index(i).(starlark.String); ok {
				deps.PyFiles = append(deps.PyFiles, string(s))
			}
		}
	}
	sparkJob.Spec.Deps = deps

	var lastSensorRes *spark.SensorSparkJobResponse
	totalAttempts := retryAttempts + 1
	for attempt := 1; attempt <= totalAttempts; attempt++ {
		createRes, err := r.doCreateJob(ctx, sparkJob, timeout)
		if err != nil {
			return nil, err
		}

		sensorRes, err := r.doSensorJob(ctx, createRes.SparkJob, timeout, pollSeconds, utils.SucceededCondition)
		if err != nil {
			return nil, err
		}
		lastSensorRes = sensorRes

		if isSparkJobSucceeded(sensorRes.SparkJob) {
			break
		}

		if isSparkJobKilled(sensorRes.SparkJob) {
			logger.Error("spark job killed, no retry", ext.ZapError(fmt.Errorf("spark job killed"))...)
			break
		}

		if attempt < totalAttempts {
			logger.Info(fmt.Sprintf("spark job failed (attempt %d/%d), retrying", attempt, totalAttempts))
		} else {
			logger.Error(fmt.Sprintf("spark job failed after all %d attempts", totalAttempts))
		}
	}

	var sparkJobValue starlark.Value
	if err := utils.AsStar(lastSensorRes.SparkJob, &sparkJobValue); err != nil {
		logger.Error(errorReasonConvertStarlarkValue, ext.ZapError(err)...)
		return nil, err
	}
	return sparkJobValue, nil
}

func isSparkJobSucceeded(job *v2pb.SparkJob) bool {
	if job == nil {
		return false
	}
	for _, c := range job.Status.GetStatusConditions() {
		if c.Type == utils.SucceededCondition && c.Status == apipb.CONDITION_STATUS_TRUE {
			return true
		}
	}
	return false
}

func isSparkJobKilled(job *v2pb.SparkJob) bool {
	if job == nil {
		return false
	}
	for _, c := range job.Status.GetStatusConditions() {
		if c.Type == utils.KilledCondition && c.Status == apipb.CONDITION_STATUS_TRUE {
			return true
		}
	}
	return false
}

func getRunningConditionType(receiver starlark.Value) (starlark.Value, error) {
	return starlark.String(utils.SparkAppRunningCondition), nil
}

func getSucceededConditionType(receiver starlark.Value) (starlark.Value, error) {
	return starlark.String(utils.SucceededCondition), nil
}

func getKilledConditionType(receiver starlark.Value) (starlark.Value, error) {
	return starlark.String(utils.KilledCondition), nil
}
