package com.example.demo.domain;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import java.time.LocalDateTime;

@Entity
@Getter @Setter
public class Incident {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(columnDefinition = "TEXT")
    private String content;

    private String keywords;      // AI 추출 키워드
    private String visionResults; // YOLO 분석 결과
    private boolean verified;     // [필수 추가] 일치 여부 필드

    private LocalDateTime createdAt = LocalDateTime.now();
}