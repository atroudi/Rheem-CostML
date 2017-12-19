# Rheem-CostML
This component is a blackbox Machine leaning module for rheem that will help to learn from profiling logs the cost in an offline mode.
The learning model will be based on learning rheem core cost model on exhaustively generated rheem plans. 
PS: Log generation is ensured by the `rheem-profiler` module.

# Overview
The contained ML technics currently used are: Forest, Neural networks,. 
All technics are running on keras ontop of tensorFlow.

## How to Learn Rheem Cost model
Learning offline Rheem core model steps:
1. Generate Rheem plan logs with Rheem-profiler
2. Learn rheem-ML model
3. Enable rheem optimizer to use rheem-ML  
