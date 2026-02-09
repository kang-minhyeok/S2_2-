package com.example.demo.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import java.util.List;

@Getter
@NoArgsConstructor
public class VisionResponse {
    private String status;
    private int detected_count;
    private List<DetectedObject> objects;

    @Getter
    @NoArgsConstructor
    public static class DetectedObject {
        private String label;      // 탐지된 물체 (예: person, knife)
        private double confidence; // 신뢰도
    }
}