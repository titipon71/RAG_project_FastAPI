// index.js

// ยิง GET request ไปที่ root "/"
fetch("https://3e5e47263a9d.ngrok-free.app/") // แก้ port/URL ให้ตรงกับ API ของคุณ
  .then(response => {
    if (!response.ok) {
      throw new Error("HTTP error! Status: " + response.status);
    }
    return response.json(); // ถ้า API ส่ง JSON มาใช้ .json()
  })
  .then(data => {
    console.log("ผลลัพธ์จาก API:");
    console.log(data);
  })
  .catch(error => {
    console.error("เกิดข้อผิดพลาด:", error.message);
  });
