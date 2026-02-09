package com.example.demo.service;

import com.example.demo.domain.Incident;
import com.example.demo.repository.IncidentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class AiService {
    private final WebClient webClient;
    private final IncidentRepository incidentRepository;

    public AiService(IncidentRepository incidentRepository) {
        this.incidentRepository = incidentRepository;
        this.webClient = WebClient.builder().baseUrl("http://127.0.0.1:8000").build();
    }

    public void processUnifiedReport(String content, MultipartFile file) {
        log.info("통합 분석 및 저장 시작...");
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", file.getResource());

        try {
            Map<String, Object> response = webClient.post()
                    .uri(uriBuilder -> uriBuilder.path("/analyze/unified").queryParam("content", content).build())
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve().bodyToMono(Map.class).block();

            if (response != null) {
                Incident incident = new Incident();
                incident.setContent(content);

                // 1. 키워드 추출 안전하게 처리 (null 방지)
                Object keywordsObj = response.get("keywords");
                if (keywordsObj instanceof List) {
                    List<?> list = (List<?>) keywordsObj;
                    String keywordsStr = list.stream().map(Object::toString).collect(Collectors.joining(", "));
                    incident.setKeywords(keywordsStr);
                }

                // 2. 비전 결과 추출 (끝에 쉼표 남지 않게 수정)
                Object visionObj = response.get("vision_results");
                if (visionObj instanceof List) {
                    List<Map<String, Object>> visionList = (List<Map<String, Object>>) visionObj;
                    String visionStr = visionList.stream()
                            .map(obj -> {
                                String color = (String) obj.get("color");
                                return obj.get("label") + (color.isEmpty() ? "" : "(" + color + ")");
                            })
                            .collect(Collectors.joining(", ")); // 쉼표 문제를 해결합니다
                    incident.setVisionResults(visionStr);
                }

                // 3. 검증 여부 저장
                Boolean isVerified = (Boolean) response.get("is_verified");
                incident.setVerified(isVerified != null && isVerified);

                incidentRepository.save(incident);
                log.info("DB 저장 완료: {}", incident.getKeywords());
            }
        } catch (Exception e) { log.error("통합 처리 오류: {}", e.getMessage()); }
    }

    // 컨트롤러 에러 방지용 메서드
    public void processReport(String content) { log.info("단순 텍스트 처리: {}", content); }
}