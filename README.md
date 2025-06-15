Heap Data Structures Implementation

A comprehensive implementation of heap data structures featuring Heapsort algorithm and Priority Queue with task scheduling.

Summary

This project implements and analyzes heap-based algorithms:
- Heapsort: O(n log n) sorting algorithm with guaranteed performance
- Priority Queue: Binary heap with O(log n) operations and position mapping optimization
- Task Scheduler: Real-world application for dynamic task management

## Prerequisites

```bash
pip install matplotlib
```

## Usage

### Run Heapsort Analysis
```bash
python heapSort.py
```
Tests heapsort performance against QuickSort and MergeSort across different input types and sizes.

### Run Priority Queue Demo
```bash
python priorityQueue.py
```
Demonstrates priority queue operations, task scheduling, and performance benchmarking.

Files

- `heapSort.py` - Heapsort implementation and benchmarking
- `priorityQueue.py` - Priority queue and task scheduler

Key Features

- Array-based binary heap implementation
- In-place sorting with O(1) space complexity
- Position mapping for O(log n) key updates
- Comprehensive performance validation
- Real-time task scheduling capabilities
