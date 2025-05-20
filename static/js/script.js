// script.js - Xử lý tương tác cho ứng dụng phát hiện trùng lặp văn bản tiếng Việt

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('duplicate-form');
    const fileForm = document.getElementById('file-form');
    const resultContainer = document.getElementById('result-container');
    const resultSummary = document.getElementById('result-summary');
    const similarityResults = document.getElementById('similarity-results');
    const modelResults = document.getElementById('model-results');
    const highlightedTextA = document.getElementById('highlighted-text-a');
    const highlightedTextB = document.getElementById('highlighted-text-b');
    const detailedSummary = document.getElementById('detailed-summary');
    const similarityMarker = document.getElementById('similarity-marker');
    const btnFeedback = document.getElementById('btn-feedback');
    const btnExport = document.getElementById('btn-export');
    
    // Biến lưu biểu đồ
    let similarityChart = null;
    
    // Khởi tạo tooltips bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Cài đặt sự kiện cho nút feedback
    if (btnFeedback) {
        btnFeedback.addEventListener('click', function() {
            const feedbackModal = new bootstrap.Modal(document.getElementById('feedbackModal'));
            feedbackModal.show();
        });
    }
    
    // Cài đặt sự kiện cho nút xuất PDF
    if (btnExport) {
        btnExport.addEventListener('click', function() {
            exportToPDF();
        });
    }
    
    // Form kiểm tra văn bản nhập trực tiếp
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Hiển thị nút đang xử lý
            const submitBtn = document.getElementById('check-btn');
            const originalBtnText = submitBtn.textContent;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Đang kiểm tra...';
            submitBtn.disabled = true;
            
            // Lấy dữ liệu
            const formData = new FormData(form);
            
            try {
                // Gửi yêu cầu
                const response = await fetch('/check_duplicate', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Hiển thị kết quả
                displayResults(data);
                
            } catch (error) {
                showError('Đã xảy ra lỗi khi xử lý yêu cầu: ' + error.message);
            } finally {
                // Khôi phục nút gửi
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
    
    // Form kiểm tra bằng file
    if (fileForm) {
        fileForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Hiển thị nút đang xử lý
            const submitBtn = document.getElementById('check-file-btn');
            const originalBtnText = submitBtn.textContent;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Đang kiểm tra...';
            submitBtn.disabled = true;
            
            // Lấy dữ liệu file
            const formData = new FormData(fileForm);
            
            try {
                // Gửi yêu cầu
                const response = await fetch('/check_duplicate_files', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                    return;
                }
                
                // Hiển thị kết quả
                displayResults(data);
                
            } catch (error) {
                showError('Đã xảy ra lỗi khi xử lý file: ' + error.message);
            } finally {
                // Khôi phục nút gửi
                submitBtn.innerHTML = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
    
    function displayResults(data) {
        // Xóa kết quả cũ
        similarityResults.innerHTML = '';
        modelResults.innerHTML = '';
        
        // Lấy văn bản từ form hoặc từ dữ liệu trả về (nếu tải file)
        const textA = data.text_a || document.getElementById('text_a').value;
        const textB = data.text_b || document.getElementById('text_b').value;
        
        // Xử lý và hiển thị độ tương đồng
        const features = data.features;
        
        // Xác định kết quả tổng hợp dựa trên dự đoán của đa số mô hình
        let trueCount = 0;
        let totalModels = 0;
        
        for (const modelName in data.predictions) {
            if (data.predictions[modelName].prediction === 1) {
                trueCount++;
            }
            totalModels++;
        }
        
        const majorityIsDuplicate = trueCount > totalModels / 2;
        const averageProbability = calculateAverageProbability(data.predictions);
        
        // Tính toán mức độ trùng lặp tổng thể dựa trên các chỉ số
        const overallSimilarity = calculateOverallSimilarity(features);
        
        // Thiết lập thanh đo mức độ tương đồng
        updateSimilarityMarker(overallSimilarity);
        
        // Thiết lập thông báo tóm tắt
        if (majorityIsDuplicate) {
            resultSummary.className = 'alert alert-danger';
            resultSummary.innerHTML = `<strong>Phát hiện trùng lặp!</strong> Đa số các mô hình dự đoán rằng hai văn bản trên có khả năng cao là trùng lặp. <strong>Mức độ trùng lặp tổng thể: ${(overallSimilarity * 100).toFixed(2)}%</strong>`;
        } else {
            resultSummary.className = 'alert alert-success';
            resultSummary.innerHTML = `<strong>Không phát hiện trùng lặp!</strong> Đa số các mô hình dự đoán rằng hai văn bản không trùng lặp. <strong>Mức độ trùng lặp tổng thể: ${(overallSimilarity * 100).toFixed(2)}%</strong>`;
        }
        
        // Hiển thị các độ tương đồng
        displaySimilarity('Cosine Similarity', features.cosine);
        displaySimilarity('Jaccard Similarity', features.jaccard);
        displaySimilarity('Tỷ lệ từ chung', features.common_ratio);
        displaySimilarity('Tỷ lệ độ dài', features.len_ratio);
        
        // Hiển thị kết quả từ các mô hình
        for (const [modelName, result] of Object.entries(data.predictions)) {
            const displayName = getDisplayModelName(modelName);
            const row = document.createElement('tr');
            
            const isDuplicate = result.prediction === 1;
            const resultClass = isDuplicate ? 'model-result-positive' : 'model-result-negative';
            const resultText = isDuplicate ? 'Trùng lặp' : 'Không trùng lặp';
            
            let probabilityText = '';
            if (result.probability !== null) {
                // Hiển thị 4 chữ số thập phân thay vì 2 để tránh làm tròn xuống 0%
                const probability = (result.probability * 100);
                // Nếu xác suất rất nhỏ (dưới 0.01%) thì hiển thị < 0.01% thay vì 0.00%
                if (probability < 0.01 && probability > 0) {
                    probabilityText = ` (< 0.01%)`;
                } else {
                    probabilityText = ` (${probability.toFixed(4)}%)`;
                }
            }
            
            row.innerHTML = `
                <td>${displayName}</td>
                <td class="${resultClass}">${resultText}${probabilityText}</td>
            `;
            
            modelResults.appendChild(row);
        }
        
        // Vẽ biểu đồ cột
        drawSimilarityChart(features);
        
        // Hiển thị đánh dấu từ chung và khác biệt
        highlightCommonWords(textA, textB);
        
        // Tạo tóm tắt chi tiết
        createDetailedSummary(textA, textB, features, majorityIsDuplicate, averageProbability);
        
        // Hiển thị kết quả container
        resultContainer.style.display = 'block';
        
        // Cuộn xuống kết quả
        resultContainer.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Tính toán mức độ trùng lặp tổng thể từ các chỉ số
    function calculateOverallSimilarity(features) {
        // Tính trung bình có trọng số của các chỉ số
        // Cosine và Jaccard có trọng số cao hơn vì chúng phản ánh tốt hơn về ngữ nghĩa và từ vựng
        const weights = {
            cosine: 0.4,      // 40% trọng số cho Cosine (ngữ nghĩa)
            jaccard: 0.3,     // 30% trọng số cho Jaccard (từ vựng)
            common_ratio: 0.2, // 20% trọng số cho tỷ lệ từ chung
            len_ratio: 0.1     // 10% trọng số cho tỷ lệ độ dài
        };
        
        const weightedSum = 
            features.cosine * weights.cosine +
            features.jaccard * weights.jaccard +
            features.common_ratio * weights.common_ratio +
            features.len_ratio * weights.len_ratio;
            
        return weightedSum;
    }
    
    function updateSimilarityMarker(similarity) {
        const percentage = similarity * 100;
        const position = Math.min(Math.max(percentage, 0), 100);
        
        if (similarityMarker) {
            similarityMarker.style.left = `${position}%`;
            
            // Thêm tooltip hiển thị giá trị cụ thể
            similarityMarker.setAttribute('title', `Mức độ trùng lặp: ${percentage.toFixed(2)}%`);
            
            // Thêm hiệu ứng nhấp nháy để thu hút sự chú ý
            similarityMarker.classList.add('blink-animation');
            setTimeout(() => {
                similarityMarker.classList.remove('blink-animation');
            }, 1000);
        }
    }
    
    function calculateAverageProbability(predictions) {
        let sum = 0;
        let count = 0;
        
        for (const modelName in predictions) {
            if (predictions[modelName].probability !== null) {
                sum += predictions[modelName].probability;
                count++;
            }
        }
        
        return count > 0 ? sum / count : 0;
    }
    
    function drawSimilarityChart(features) {
        const chartCanvas = document.getElementById('similarityChart');
        if (!chartCanvas) return;
        
        // Nếu đã có biểu đồ trước đó, hủy nó
        if (similarityChart) {
            similarityChart.destroy();
        }
        
        // Dữ liệu cho biểu đồ
        const labels = ['Cosine', 'Jaccard', 'Tỷ lệ từ chung', 'Tỷ lệ độ dài'];
        const values = [
            features.cosine * 100,
            features.jaccard * 100,
            features.common_ratio * 100,
            features.len_ratio * 100
        ];
        
        // Sử dụng màu nhẹ nhàng hơn (pastel) cho mỗi chỉ số
        const backgroundColors = [
            '#66c2a5', // Mint pastel cho Cosine
            '#fc8d62', // Cam nhạt cho Jaccard
            '#8da0cb', // Xanh dương pastel cho Tỷ lệ từ chung
            '#e78ac3'  // Hồng pastel cho Tỷ lệ độ dài
        ];
        
        // Tạo biểu đồ
        similarityChart = new Chart(chartCanvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Tỷ lệ tương đồng (%)',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Tỷ lệ %'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Chỉ số phân tích'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false // Ẩn legend vì mỗi cột có một màu khác nhau
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.parsed.y.toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function highlightCommonWords(textA, textB) {
        if (!highlightedTextA || !highlightedTextB) return;
        
        // Tách văn bản thành các từ
        const wordsA = textA.split(/\s+/);
        const wordsB = textB.split(/\s+/);
        
        // Tạo tập hợp từ để xác định từ chung
        const setA = new Set(wordsA);
        const setB = new Set(wordsB);
        const commonWords = new Set([...setA].filter(word => setB.has(word)));
        
        // Tạo HTML với từ được đánh dấu
        const htmlA = wordsA.map(word => 
            commonWords.has(word) 
                ? `<span class="common-word">${word}</span>` 
                : `<span class="different-word">${word}</span>`
        ).join(' ');
        
        const htmlB = wordsB.map(word => 
            commonWords.has(word) 
                ? `<span class="common-word">${word}</span>` 
                : `<span class="different-word">${word}</span>`
        ).join(' ');
        
        // Cập nhật nội dung
        highlightedTextA.innerHTML = htmlA;
        highlightedTextB.innerHTML = htmlB;
        
        // Cập nhật mẫu màu
        document.querySelector('.common-word-sample').className = 'common-word-sample common-word';
        document.querySelector('.different-word-sample').className = 'different-word-sample different-word';
    }
    
    function createDetailedSummary(textA, textB, features, isDuplicate, averageProbability) {
        if (!detailedSummary) return;
        
        const wordsA = textA.split(/\s+/).length;
        const wordsB = textB.split(/\s+/).length;
        const totalWords = wordsA + wordsB;
        
        // Tính toán số từ chung dựa trên tỷ lệ từ chung
        const estimatedCommonWords = Math.round(features.common_ratio * Math.min(wordsA, wordsB));
        
        // Tính toán mức độ trùng lặp tổng thể
        const overallSimilarity = calculateOverallSimilarity(features);
        
        // Tạo tóm tắt
        let summary = `
            <strong>Phân tích chi tiết:</strong> Hai văn bản có tổng cộng ${totalWords} từ (${wordsA} từ trong văn bản 1 và ${wordsB} từ trong văn bản 2). 
            Có khoảng ${estimatedCommonWords} từ xuất hiện trong cả hai văn bản.
        `;
        
        // Tóm tắt các chỉ số
        summary += `<div class="mt-2"><strong>Chỉ số quan trọng:</strong>
            <ul>
                <li>Độ tương đồng về mặt ngữ nghĩa (Cosine): <strong>${(features.cosine * 100).toFixed(2)}%</strong></li>
                <li>Độ tương đồng về từ vựng (Jaccard): <strong>${(features.jaccard * 100).toFixed(2)}%</strong></li>
                <li>Mức độ trùng lặp tổng thể: <strong>${(overallSimilarity * 100).toFixed(2)}%</strong></li>
            </ul></div>`;
        
        // Đánh giá và gợi ý hành động
        if (isDuplicate) {
            const confidenceLevel = (averageProbability * 100).toFixed(2);
            const severityLevel = getSeverityLevel(features);
            
            summary += `<div class="alert alert-danger mt-2">
                <strong>Kết luận:</strong> Phát hiện trùng lặp với độ tin cậy ${confidenceLevel}%.
                <div class="mt-1"><strong>Mức độ:</strong> ${severityLevel.text}</div>
                <div class="mt-2"><strong>Gợi ý hành động:</strong></div>
                <ul class="mb-0">
                    ${severityLevel.actions.map(action => `<li>${action}</li>`).join('')}
                </ul>
            </div>`;
        } else {
            const confidenceLevel = (100 - (averageProbability * 100)).toFixed(2);
            
            summary += `<div class="alert alert-success mt-2">
                <strong>Kết luận:</strong> Không phát hiện trùng lặp với độ tin cậy ${confidenceLevel}%.
                <div class="mt-2"><strong>Gợi ý hành động:</strong></div>
                <ul class="mb-0">
                    <li>Nội dung có thể được coi là nguyên bản</li>
                    <li>Vẫn nên kiểm tra nguồn và trích dẫn nếu sử dụng ý tưởng từ tài liệu khác</li>
                </ul>
            </div>`;
        }
        
        detailedSummary.innerHTML = summary;
    }
    
    // Hàm xác định mức độ nghiêm trọng và gợi ý hành động
    function getSeverityLevel(features) {
        const cosine = features.cosine;
        const jaccard = features.jaccard;
        const average = (cosine + jaccard) / 2;
        
        if (average >= 0.8) {
            return {
                text: "Rất nghiêm trọng - Trùng lặp rõ ràng",
                actions: [
                    "Viết lại hoàn toàn nội dung với từ ngữ và cấu trúc riêng của bạn",
                    "Trích dẫn rõ ràng nguồn gốc và đặt trong dấu ngoặc kép nếu muốn sử dụng nguyên văn",
                    "Cân nhắc sử dụng công cụ viết lại để tạo nội dung mới"
                ]
            };
        } else if (average >= 0.6) {
            return {
                text: "Nghiêm trọng - Trùng lặp đáng kể",
                actions: [
                    "Viết lại các phần trùng lặp với từ ngữ riêng của bạn",
                    "Đảm bảo trích dẫn nguồn khi sử dụng ý tưởng từ văn bản gốc",
                    "Thêm phân tích hoặc ý kiến của riêng bạn để làm phong phú nội dung"
                ]
            };
        } else if (average >= 0.4) {
            return {
                text: "Trung bình - Tương đồng đáng chú ý",
                actions: [
                    "Xem xét lại những phần tương đồng",
                    "Bổ sung thêm thông tin và góc nhìn mới",
                    "Đảm bảo ghi nhận nguồn tham khảo"
                ]
            };
        } else {
            return {
                text: "Nhẹ - Có một số điểm tương đồng",
                actions: [
                    "Có thể tiếp tục sử dụng với một số điều chỉnh nhỏ",
                    "Nên ghi chú nguồn tham khảo nếu có sử dụng ý tưởng từ văn bản khác"
                ]
            };
        }
    }
    
    function displaySimilarity(name, value) {
        const row = document.createElement('tr');
        const percentage = (value * 100).toFixed(2);
        
        let colorClass = 'similarity-low';
        if (value >= 0.7) {
            colorClass = 'similarity-high';
        } else if (value >= 0.4) {
            colorClass = 'similarity-medium';
        }
        
        row.innerHTML = `
            <td>${name}</td>
            <td>
                <div class="${colorClass} percentage-value">${percentage}%</div>
                <div class="progress mt-1" style="height: 5px;">
                    <div class="progress-bar bg-${colorClass.split('-')[1]}" role="progressbar" 
                         style="width: ${percentage}%;" aria-valuenow="${percentage}" 
                         aria-valuemin="0" aria-valuemax="100"></div>
                </div>
            </td>
        `;
        
        similarityResults.appendChild(row);
    }
    
    function getDisplayModelName(name) {
        switch (name) {
            case 'knn':
                return 'K-Nearest Neighbors';
            case 'naive_bayes':
                return 'Naive Bayes';
            case 'decision_tree':
                return 'Decision Tree';
            default:
                return name.charAt(0).toUpperCase() + name.slice(1).replace('_', ' ');
        }
    }
    
    function showError(message) {
        resultSummary.className = 'alert alert-warning';
        resultSummary.textContent = message;
        resultContainer.style.display = 'block';
        
        // Xóa các kết quả chi tiết
        similarityResults.innerHTML = '';
        modelResults.innerHTML = '';
        
        // Nếu có biểu đồ, hủy nó
        if (similarityChart) {
            similarityChart.destroy();
            similarityChart = null;
        }
        
        if (highlightedTextA) highlightedTextA.innerHTML = '';
        if (highlightedTextB) highlightedTextB.innerHTML = '';
        if (detailedSummary) detailedSummary.innerHTML = '';
    }
    
    function exportToPDF() {
        // Sử dụng thư viện jsPDF và html2canvas
        const { jsPDF } = window.jspdf;
        
        // Tạo tên file
        const filename = `Báo cáo trùng lặp - ${new Date().toLocaleString('vi-VN')}.pdf`;
        
        // Lấy nội dung cần xuất
        const element = document.getElementById('result-container');
        
        // Thông báo cho người dùng
        btnExport.disabled = true;
        btnExport.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Đang tạo PDF...';
        
        // Tạo PDF
        html2canvas(element, {
            scale: 1.5,
            useCORS: true,
            logging: false
        }).then(canvas => {
            const imgData = canvas.toDataURL('image/png');
            const pdf = new jsPDF('p', 'mm', 'a4');
            const imgProps = pdf.getImageProperties(imgData);
            const pdfWidth = pdf.internal.pageSize.getWidth();
            const pdfHeight = (imgProps.height * pdfWidth) / imgProps.width;
            
            // Thêm tiêu đề
            pdf.setFontSize(18);
            pdf.text('Báo Cáo Kiểm Tra Văn Bản Trùng Lặp', pdfWidth/2, 15, { align: 'center' });
            
            // Thêm thời gian
            pdf.setFontSize(12);
            pdf.text(`Được tạo vào: ${new Date().toLocaleString('vi-VN')}`, pdfWidth/2, 25, { align: 'center' });
            
            // Thêm ảnh kết quả
            pdf.addImage(imgData, 'PNG', 0, 35, pdfWidth, pdfHeight);
            
            // Lưu file
            pdf.save(filename);
            
            // Khôi phục nút
            btnExport.disabled = false;
            btnExport.innerHTML = '<i class="bi bi-download"></i> Xuất báo cáo PDF';
        });
    }
}); 