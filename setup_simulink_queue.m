% SETUP_SIMULINK_QUEUE Programmatic Creation & Configuration of Telemedicine Queue Model
%
% Project: Explainable AI for DR Screening in Rural India (SIH 2026, PS ID 26038)
% Module 5: Simulink Telemedicine Workflow Simulation
%
% Description:
%   Programmatically constructs a Simulink discrete-event queueing model ('telemedicine_queue.slx')
%   simulating rural clinic image ingestion, 2 Mbps network upload bottleneck, server AI processing
%   throughput (Modules 1-4), and remote doctor review queues for 100,000+ annual patient volume.

modelName = 'telemedicine_queue';

% Close model if already open
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

% Create new Simulink model
new_system(modelName);
open_system(modelName);

% Set simulation parameters (1-day 86400 seconds or 1-year 3.1536e7 seconds)
set_param(modelName, 'StopTime', '86400'); % 24 hours simulation
set_param(modelName, 'Solver', 'FixedStepDiscrete', 'FixedStep', '1.0');

% Add Blocks:
% 1. Clinic Patient Arrival Signal (Pulse Generator)
add_block('simulink/Sources/Pulse Generator', [modelName '/ClinicArrivalGenerator'], ...
    'PulseType', 'Time based', ...
    'Period', '180', ...          % Arrival every 3 mins (20 patients/hr per clinic)
    'PulseWidth', '50', ...
    'Amplitude', '2.5');         % 2.5 MB image size per arrival

% 2. Bandwidth Bottleneck Subsystem (2 Mbps Rural Link vs 15 Mbps 4G)
add_block('simulink/Math Operations/Gain', [modelName '/UploadBandwidthLimit'], ...
    'Gain', '0.25', ...          % Delay scaling for 2 Mbps link (2.5 MB / (0.25 MB/s) = 10s upload)
    'Position', [180, 50, 240, 90]);

% 3. Server AI Execution Delay (Modules 1-4: ~1.2s per image)
add_block('simulink/Math Operations/Gain', [modelName '/AIServerProcessingDelay'], ...
    'Gain', '1.2', ...
    'Position', [300, 50, 360, 90]);

% 4. Automated Triage Gate (Pass-through for clear Level 0 cases: ~60% pass-through)
add_block('simulink/Math Operations/Gain', [modelName '/AutomatedTriagePassThrough'], ...
    'Gain', '0.40', ...          % Only 40% routed to human doctor queue
    'Position', [420, 50, 480, 90]);

% 5. Doctor Review Queue Integrator (Queue Length Over Time)
add_block('simulink/Continuous/Integrator', [modelName '/DoctorReviewQueueIntegrator'], ...
    'InitialCondition', '0', ...
    'Position', [540, 50, 590, 90]);

% 6. Doctor Review Capacity Subsystem (Doctors x 30s review rate)
add_block('simulink/Sources/Constant', [modelName '/DoctorServiceCapacity'], ...
    'Value', '0.0333', ...       % 1 case per 30s = 0.0333 cases/sec per doctor
    'Position', [540, 140, 590, 170]);

% 7. Scope for Visual Real-Time Dashboard Output
add_block('simulink/Sinks/Scope', [modelName '/QueueLengthScope'], ...
    'Position', [660, 50, 710, 90]);

% Connect Blocks
add_line(modelName, 'ClinicArrivalGenerator/1', 'UploadBandwidthLimit/1');
add_line(modelName, 'UploadBandwidthLimit/1', 'AIServerProcessingDelay/1');
add_line(modelName, 'AIServerProcessingDelay/1', 'AutomatedTriagePassThrough/1');
add_line(modelName, 'AutomatedTriagePassThrough/1', 'DoctorReviewQueueIntegrator/1');
add_line(modelName, 'DoctorReviewQueueIntegrator/1', 'QueueLengthScope/1');

% Save model
save_system(modelName, [modelName '.slx']);
fprintf('Simulink telemedicine queue model successfully built and saved as "%s.slx"\n', modelName);
