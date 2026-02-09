package com.example.demo.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class KeywordResponse {
    private String status;
    private String original_content;
    private List<String> extracted_keywords; // 추출된 명사 리스트
}