// window.onload = function () {
//     document.getElementById("download")
//         .addEventListener("click", () => {
//             var report = document.getElementById("report");
//             // console.log(report);
//             // console.log(window);
//             var opt = {
                
//                 filename: 'myreport1.pdf',
//                 image: { type: 'jpeg', quality: 0.98 },
//                 html2canvas: { scale: 2 },
//                 jsPDF: { unit: 'in', format: 'letter', orientation: 'landscape' }
//             };
//             html2pdf().from(report).set(opt).save();
//         })
// }

function generatePDF(){
    const report = document.getElementById("report");
    html2pdf().from(report).save();
}