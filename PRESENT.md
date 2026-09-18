# Hackathon Presentation Brief

## Slide 1: Title
- Dynamic AI Traffic Flow Optimizer & Emergency Grid
- Real-time traffic control using camera vision and emergency priority
- Goal: reduce congestion and give emergency vehicles a clear path

## Slide 2: Problem
- Fixed traffic lights waste time during low and high traffic changes
- Emergency vehicles lose valuable seconds at intersections
- Manual traffic control does not react fast enough to live conditions

## Slide 3: Proposed Solution
- Use YOLO-based vehicle detection from live video
- Count vehicles per camera approach
- Compute congestion scores instead of only raw counts
- Change signal timings dynamically based on live traffic

## Slide 4: Emergency Handling
- Detect emergency vehicles such as ambulances and fire trucks
- Support manual emergency trigger with key `e`
- Apply corridor-lane priority instead of unsafe full green
- Show emergency source clearly in the UI and logs

## Slide 5: System Architecture
- Data acquisition: video frame input and detection
- Processing: lane counting, congestion scoring, signal decision
- Execution: signal state output, overlay, and metrics logging
- Design is modular and easy to explain in a demo

## Slide 6: Demo and Evidence
- Live HUD shows counts, queue, score, signal state, and emergency status
- Benchmark compares baseline vs adaptive control
- Judge demo runner creates video, metrics, report, and orchestrator artifact
- SUMO is available as an optional simulation layer

## Slide 7: Results and Value
- Adaptive logic reacts to traffic instead of fixed timing
- Emergency vehicles get priority with explainable control
- The system is measurable, visible, and hackathon-ready
- Prototype is practical for a real camera-based intersection

## Slide 8: Honest Scope
- Live runtime: single intersection
- Multi-node corridor: prototype pre-clear simulation
- SUMO: optional validation/demo layer
- Focus: correctness, clarity, and judge-friendly presentation
